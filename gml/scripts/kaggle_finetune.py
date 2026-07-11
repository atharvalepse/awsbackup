"""Kaggle-optimized FT-2: LoRA fine-tune Qwen2.5-3B-Instruct on LOCOMO QA.

Paste this whole file into one Kaggle notebook cell and hit Run All.

Kaggle setup before running:
  - Notebook → Settings → Internet: ON
  - Accelerator: GPU T4 x1   (P100 also works, slower)
  - No Kaggle dataset upload needed; LOCOMO source is fetched from GitHub

Why this script exists alongside scripts/finetune_answer_llm.py:
  The local script uses bf16 compute (Ampere+). Kaggle's T4 is Turing (sm_75),
  fp16 only — and P100 is Pascal (sm_60), also fp16. So we force fp16 throughout.

Output:
  /kaggle/working/qwen_locomo_ft/adapter/   ← PEFT adapter (~25 MB)
  /kaggle/working/qwen_locomo_ft_adapter.zip ← zipped for one-click download
"""
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path


# ---- Config -----------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_out"
OUT_DIR = WORK_DIR / "qwen_locomo_ft"
DATA_DIR = WORK_DIR / "locomo_ft_data"

# Same 8-train / 2-test conversation split as scripts/prepare_ft_llm_data.py
# (holds out conv-49 and conv-50 for downstream eval).
TRAIN_CONV_IDS = set(range(8))

# Training hyperparams. With 16 GB on T4 we don't need to shrink — match the
# transcript's local plan (r=16, max-len=1024) instead of the 4 GB compromise.
EPOCHS = 2
BATCH = 2
GRAD_ACCUM = 4
LORA_R = 16
LORA_ALPHA = 32
MAX_LEN = 1024
LR = 2e-4

# Pip installs Kaggle's base image is missing or has outdated.
PIP_INSTALL = "peft>=0.10 trl>=0.8 bitsandbytes>=0.43 transformers>=4.40 accelerate>=0.30 datasets>=2.18"


# ---- 0. Install deps --------------------------------------------------------
print("[0/6] installing dependencies (silent)...")
os.system(f"pip install -q --upgrade {PIP_INSTALL}")


# ---- 1. Download LOCOMO -----------------------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

locomo_path = DATA_DIR / "locomo10.json"
if not locomo_path.exists():
    print(f"[1/6] downloading LOCOMO from {LOCOMO_URL}")
    urllib.request.urlretrieve(LOCOMO_URL, locomo_path)
print(f"  LOCOMO: {locomo_path.stat().st_size / 1e6:.1f} MB")


# ---- 2. Prep FT data (inlined from scripts/prepare_ft_llm_data.py) ---------
def _index_conversation(conv: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    block = conv.get("conversation", conv)
    if not isinstance(block, dict):
        return out
    for key, messages in block.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(messages, list):
            continue
        for m in messages:
            if not isinstance(m, dict):
                continue
            did = m.get("dia_id")
            text = m.get("text") or m.get("content") or ""
            if did and text:
                speaker = m.get("speaker") or ""
                out[did] = f"{speaker}: {text}" if speaker else text
    return out


def _build_example(qa: dict, conv_idx: dict[str, str]) -> dict | None:
    q = qa.get("question", "").strip()
    if not q:
        return None
    gold = qa.get("answer") or qa.get("adversarial_answer") or ""
    if not gold:
        return None
    evidence_ids = qa.get("evidence") or []
    evidence_texts = [
        conv_idx[eid] for eid in evidence_ids
        if isinstance(eid, str) and eid in conv_idx
    ]
    if not evidence_texts:
        return None
    context = "\n".join(f"<memory>{t}</memory>" for t in evidence_texts)
    prompt = (
        "Below is some context from a long-running conversation, followed by "
        "a question. Answer the question concisely.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {q}\n\nANSWER:"
    )
    return {
        "prompt": prompt,
        "completion": str(gold),
        "category": qa.get("category"),
    }


print("[2/6] preparing FT examples...")
locomo = json.loads(locomo_path.read_text(encoding="utf-8"))
train_examples: list[dict] = []
test_examples: list[dict] = []
for i, conv in enumerate(locomo):
    conv_idx = _index_conversation(conv)
    target = train_examples if i in TRAIN_CONV_IDS else test_examples
    for qa in conv.get("qa", []):
        ex = _build_example(qa, conv_idx)
        if ex is not None:
            target.append(ex)

with (DATA_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
    for ex in train_examples:
        f.write(json.dumps(ex) + "\n")
with (DATA_DIR / "test.jsonl").open("w", encoding="utf-8") as f:
    for ex in test_examples:
        f.write(json.dumps(ex) + "\n")

from collections import Counter
cat_counts = Counter(ex["category"] for ex in train_examples if ex.get("category") is not None)
print(f"  {len(train_examples)} train / {len(test_examples)} test")
print(f"  train categories: {dict(sorted(cat_counts.items()))}")


# ---- 3. Load base model in 4-bit (QLoRA) -----------------------------------
print(f"[3/6] loading {BASE_MODEL} in 4-bit (nf4 / fp16 compute)...")
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"  device: {device_name} (sm_{cap[0]}{cap[1]}), VRAM: "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# T4 (sm_75) and P100 (sm_60) lack bf16 — use fp16 compute end-to-end.
# We still use 4-bit NF4 weight quantization for a smaller memory footprint and
# the QLoRA-style paged 8-bit optimizer.
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
print(f"  base loaded in {time.time()-t0:.1f}s; "
      f"params={sum(p.numel() for p in base.parameters())/1e9:.2f}B (4-bit)")


# ---- 4. Attach LoRA --------------------------------------------------------
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()


# ---- 5. Tokenize ------------------------------------------------------------
print(f"[4/6] tokenizing {len(train_examples)} examples (max_len={MAX_LEN})...")

def _format(ex: dict) -> str:
    return f"{ex['prompt']} {ex['completion']}{tok.eos_token}"

def _tok_batch(batch):
    texts = [_format({"prompt": p, "completion": c})
             for p, c in zip(batch["prompt"], batch["completion"])]
    return tok(texts, truncation=True, max_length=MAX_LEN, padding=False)

train_ds = Dataset.from_list(train_examples)
train_ds = train_ds.map(_tok_batch, batched=True, remove_columns=train_ds.column_names)


# ---- 6. Train --------------------------------------------------------------
print(f"[5/6] training: {EPOCHS} epochs, batch={BATCH}, grad_accum={GRAD_ACCUM}, "
      f"effective_batch={BATCH*GRAD_ACCUM}, lr={LR}, lora_r={LORA_R}")

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="no",
    fp16=True,           # T4/P100 path
    bf16=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to=[],
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
t0 = time.time()
trainer.train()
elapsed_min = (time.time() - t0) / 60
print(f"\n  trained in {elapsed_min:.1f} min")


# ---- 7. Save adapter + zip -------------------------------------------------
adapter_dir = OUT_DIR / "adapter"
print(f"[6/6] saving adapter to {adapter_dir}")
model.save_pretrained(adapter_dir)
tok.save_pretrained(adapter_dir)

zip_path = WORK_DIR / "qwen_locomo_ft_adapter.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in adapter_dir.rglob("*"):
        if p.is_file():
            z.write(p, arcname=p.relative_to(adapter_dir.parent))
print(f"  zipped → {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")

# Also save a small training-config JSON so you know what produced this adapter.
(WORK_DIR / "qwen_locomo_ft_meta.json").write_text(json.dumps({
    "base_model": BASE_MODEL,
    "epochs": EPOCHS,
    "batch_per_device": BATCH,
    "grad_accum": GRAD_ACCUM,
    "effective_batch": BATCH * GRAD_ACCUM,
    "lr": LR,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "max_len": MAX_LEN,
    "n_train": len(train_examples),
    "n_test": len(test_examples),
    "train_minutes": round(elapsed_min, 1),
    "gpu": device_name,
}, indent=2))

print("\nDone. Download from the Kaggle sidebar:")
print("  - qwen_locomo_ft_adapter.zip  (the adapter, load via peft.PeftModel)")
print("  - qwen_locomo_ft_meta.json    (training config)")
