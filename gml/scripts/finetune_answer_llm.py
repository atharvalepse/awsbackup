"""LoRA fine-tune Qwen2.5-3B-Instruct on LOCOMO QA pairs.

Loads (prompt, completion) examples produced by prepare_ft_llm_data.py
and trains a small LoRA adapter (~30M params) on top of the frozen base.

Why LoRA, not full fine-tune:
  - Qwen2.5-3B is 3B params (~6GB fp16). Full FT on MPS would need
    ~24GB+ VRAM (model + grad + optimizer state + activations).
  - LoRA trains only ~30M adapter params, keeping memory ~10-12GB
    which fits in our 20GB MPS budget.

Output: a LoRA adapter directory loadable via peft.PeftModel.from_pretrained.
Use --merge to also save a fully-merged model (~3GB) ready for Ollama
import or transformers inference.

Run:
    .venv/bin/python scripts/finetune_answer_llm.py \\
        --data-dir /tmp/locomo-llm-ft-data \\
        --base Qwen/Qwen2.5-3B-Instruct \\
        --out models/qwen_locomo_ft \\
        --epochs 1 --batch 1 --grad-accum 4
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _format_example(ex: dict, eos: str) -> str:
    """Combine prompt + completion into one tokenization-ready string."""
    return f"{ex['prompt']} {ex['completion']}{eos}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max sequence length (truncate longer prompts)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train", type=int, default=0,
                        help="Cap training examples (0 = all)")
    parser.add_argument("--merge", action="store_true",
                        help="Also save a merged full-precision model")
    parser.add_argument("--use-4bit", action="store_true",
                        help="QLoRA: load base in 4-bit via bitsandbytes. "
                             "Required for 6 GB GPUs like RTX 4050. CUDA only.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 mixed-precision training (CUDA Ampere+).")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[ft-llm] device: {device}")

    # ---- Load data ---------------------------------------------------
    data_dir = Path(args.data_dir)
    train_ex = _load_jsonl(data_dir / "train.jsonl")
    test_ex = _load_jsonl(data_dir / "test.jsonl")
    if args.max_train and args.max_train < len(train_ex):
        train_ex = train_ex[: args.max_train]
    print(f"[ft-llm] {len(train_ex)} train / {len(test_ex)} test examples")

    # ---- Tokenizer ---------------------------------------------------
    print(f"[ft-llm] loading tokenizer: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token

    def _tokenize(batch):
        texts = [f"{p} {c}{eos}" for p, c in zip(batch["prompt"], batch["completion"])]
        out = tokenizer(
            texts,
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
        )
        # Causal LM: labels = input_ids (HF handles the shift internally)
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    train_ds = Dataset.from_list(train_ex).map(_tokenize, batched=True, remove_columns=["prompt", "completion", "category"])
    print(f"[ft-llm] tokenized {len(train_ds)} examples (max_len={args.max_len})")

    # ---- Model + LoRA ------------------------------------------------
    # 4-bit (QLoRA) path: only valid on CUDA. The base weights load directly
    # onto the GPU in nf4, so we MUST NOT call .to(device) afterwards
    # (would defeat the quantization and triple memory).
    use_4bit = args.use_4bit
    if use_4bit and device != "cuda":
        print(f"[ft-llm] --use-4bit requires CUDA; got device={device}. Disabling.")
        use_4bit = False

    t0 = time.time()
    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        print("[ft-llm] loading base in 4-bit (nf4 / bf16 compute)...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb_cfg,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        print(f"[ft-llm] loading base model (fp16 on {device})...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        base.to(device)
    print(f"[ft-llm] base loaded in {time.time()-t0:.1f}s; "
          f"params={sum(p.numel() for p in base.parameters())/1e9:.2f}B"
          f"{' (4-bit)' if use_4bit else ''}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ft-llm] LoRA trainable: {trainable/1e6:.1f}M params "
          f"({trainable * 100 / sum(p.numel() for p in model.parameters()):.2f}% of total)")

    # ---- Train -------------------------------------------------------
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # On CUDA we want bf16 for speed; MPS doesn't support trainer's fp16/bf16 modes.
    use_bf16 = args.bf16 or use_4bit
    if use_bf16 and device != "cuda":
        use_bf16 = False  # silently disable outside CUDA

    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",  # we save manually at the end
        fp16=False,
        bf16=use_bf16,
        gradient_checkpointing=use_4bit,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to=[],
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[ft-llm] training done in {elapsed/60:.1f} min")

    # ---- Save adapter ------------------------------------------------
    model.save_pretrained(str(out_path / "adapter"))
    tokenizer.save_pretrained(str(out_path / "adapter"))
    print(f"[ft-llm] saved LoRA adapter to {out_path / 'adapter'}")

    if args.merge:
        print("[ft-llm] merging adapter into base for full-precision export...")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_path / "merged"))
        tokenizer.save_pretrained(str(out_path / "merged"))
        print(f"[ft-llm] saved merged model to {out_path / 'merged'}")

    # ---- Summary -----------------------------------------------------
    summary = {
        "base": args.base,
        "device": device,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "train_examples": len(train_ex),
        "trainable_params_m": round(trainable / 1e6, 2),
        "training_minutes": round(elapsed / 60, 1),
        "out_path": str(out_path),
    }
    (out_path / "ft_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ft-llm] summary: {out_path / 'ft_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
