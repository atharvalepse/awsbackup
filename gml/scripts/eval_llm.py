"""Compare base vs FT'd answer LLM on token F1.

Reads the held-out LOCOMO QA test set produced by prepare_llm_ft_data
(/tmp/locomo-llm-ft-data/test.jsonl), runs each prompt through both the
base Qwen2.5-3B-Instruct and the FT'd LoRA adapter, and reports:

  - Token-F1 (exact LOCOMO metric)
  - Exact match
  - Per-category breakdown (1=single-hop, 2=multi-hop, 3=temporal,
    4=open-domain, 5=adversarial)
  - Per-example wins/losses/ties

Why we need this: FT-2 finishes with a training loss number, but that
doesn't tell us whether generated answers actually improved. This runs
the comparison directly so we know if FT-2 was worth keeping before we
wire it into the orchestration pipeline.

Run:
    .venv/bin/python scripts/eval_llm.py \\
        --base Qwen/Qwen2.5-3B-Instruct \\
        --adapter models/qwen_locomo_ft \\
        --data /tmp/locomo-llm-ft-data/test.jsonl \\
        --max-examples 100
"""
import argparse
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_NAMES = {
    1: "single-hop", 2: "multi-hop", 3: "temporal",
    4: "open-domain", 5: "adversarial",
}


def _normalize(text: str) -> str:
    """LOCOMO normalization: lowercase, strip punct/articles, collapse ws.

    Matches the standard SQuAD F1 normalizer used in the LOCOMO paper so
    our numbers are directly comparable to published baselines.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default=None,
                   help="Path to LoRA adapter dir; if None, only base is evaluated")
    p.add_argument("--data", required=True,
                   help="Path to test.jsonl with {prompt, completion, category}")
    p.add_argument("--max-examples", type=int, default=0,
                   help="Limit examples for smoke tests; 0 = all")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--device", default="mps",
                   help="mps, cuda, or cpu")
    p.add_argument("--out", default="/tmp/llm_eval_results.json")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    print(f"[eval-llm] device: {device}")

    # ---- Load data --------------------------------------------------
    examples: list[dict] = []
    with Path(args.data).open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"[eval-llm] {len(examples)} test examples")

    # ---- Load tokenizer (shared) ------------------------------------
    print(f"[eval-llm] loading tokenizer: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Helper to run one model ------------------------------------
    def run_model(name: str, model) -> list[str]:
        model.eval()
        gens: list[str] = []
        t0 = time.time()
        for i, ex in enumerate(examples):
            messages = [{"role": "user", "content": ex["prompt"]}]
            chat = tok.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            # transformers 5.x returns BatchEncoding; older versions returned a bare tensor.
            input_ids = chat["input_ids"] if hasattr(chat, "keys") else chat
            input_ids = input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen_tokens = out[0][input_ids.shape[1]:]
            text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            gens.append(text)
            if (i + 1) % 25 == 0 or i == len(examples) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                eta = (len(examples) - i - 1) / max(rate, 1e-9)
                print(f"  [{name}] {i+1}/{len(examples)} · "
                      f"{rate:.2f} ex/s · ETA {eta/60:.1f}m", flush=True)
        return gens

    # ---- Base ------------------------------------------------------
    print(f"[eval-llm] loading base: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    print(f"[eval-llm] generating base outputs...")
    base_gens = run_model("base", base_model)
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    # ---- FT'd -------------------------------------------------------
    ft_gens: list[str] = []
    if args.adapter and Path(args.adapter).exists():
        from peft import PeftModel
        print(f"[eval-llm] loading FT'd model: {args.base} + {args.adapter}")
        ft_base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        ft_model = PeftModel.from_pretrained(ft_base, args.adapter).to(device)
        print(f"[eval-llm] generating FT outputs...")
        ft_gens = run_model("ft", ft_model)
        del ft_model, ft_base
        if device == "mps":
            torch.mps.empty_cache()
    else:
        print(f"[eval-llm] no adapter at {args.adapter}; skipping FT evaluation")

    # ---- Score ------------------------------------------------------
    def score(gens: list[str]) -> dict:
        per_cat_f1: dict[int, list[float]] = defaultdict(list)
        per_cat_em: dict[int, list[float]] = defaultdict(list)
        for ex, pred in zip(examples, gens):
            cat = ex.get("category", -1)
            gold = ex["completion"]
            per_cat_f1[cat].append(token_f1(pred, gold))
            per_cat_em[cat].append(exact_match(pred, gold))
        all_f1 = [v for vs in per_cat_f1.values() for v in vs]
        all_em = [v for vs in per_cat_em.values() for v in vs]
        return {
            "overall_f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
            "overall_em": sum(all_em) / len(all_em) if all_em else 0.0,
            "per_cat_f1": {
                str(c): sum(v) / len(v) for c, v in per_cat_f1.items() if v
            },
            "per_cat_em": {
                str(c): sum(v) / len(v) for c, v in per_cat_em.items() if v
            },
            "per_cat_n": {str(c): len(v) for c, v in per_cat_f1.items()},
        }

    base_metrics = score(base_gens)
    ft_metrics = score(ft_gens) if ft_gens else None

    # ---- Report -----------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Base ({args.base})")
    print("=" * 70)
    print(f"  overall F1: {base_metrics['overall_f1']:.4f}")
    print(f"  overall EM: {base_metrics['overall_em']:.4f}")
    for c, f1 in sorted(base_metrics["per_cat_f1"].items()):
        n = base_metrics["per_cat_n"][c]
        em = base_metrics["per_cat_em"][c]
        name = CATEGORY_NAMES.get(int(c), c)
        print(f"  cat-{c} ({name}, n={n:3d}): F1={f1:.4f}  EM={em:.4f}")

    if ft_metrics:
        print()
        print("=" * 70)
        print(f"  FT'd (+ {args.adapter})")
        print("=" * 70)
        print(f"  overall F1: {ft_metrics['overall_f1']:.4f}  "
              f"(Δ {ft_metrics['overall_f1'] - base_metrics['overall_f1']:+.4f})")
        print(f"  overall EM: {ft_metrics['overall_em']:.4f}  "
              f"(Δ {ft_metrics['overall_em'] - base_metrics['overall_em']:+.4f})")
        for c, f1 in sorted(ft_metrics["per_cat_f1"].items()):
            base_f1 = base_metrics["per_cat_f1"].get(c, 0.0)
            n = ft_metrics["per_cat_n"][c]
            em = ft_metrics["per_cat_em"][c]
            name = CATEGORY_NAMES.get(int(c), c)
            print(f"  cat-{c} ({name}, n={n:3d}): F1={f1:.4f} "
                  f"(Δ {f1 - base_f1:+.4f})  EM={em:.4f}")

        # Per-example W/L/T
        wins = losses = ties = 0
        for ex, b, f in zip(examples, base_gens, ft_gens):
            bf = token_f1(b, ex["completion"])
            ff = token_f1(f, ex["completion"])
            if ff > bf + 1e-9:
                wins += 1
            elif bf > ff + 1e-9:
                losses += 1
            else:
                ties += 1
        print()
        print(f"  FT vs Base: {wins} wins · {losses} losses · {ties} ties "
              f"({wins/len(examples)*100:.1f}% / {losses/len(examples)*100:.1f}% / "
              f"{ties/len(examples)*100:.1f}%)")

    # ---- Persist ----------------------------------------------------
    payload = {
        "base": base_metrics,
        "ft": ft_metrics,
        "n_examples": len(examples),
        "base_model": args.base,
        "adapter": args.adapter,
        "samples": [
            {
                "prompt_head": ex["prompt"][:100],
                "gold": ex["completion"],
                "base_pred": b,
                "ft_pred": (ft_gens[i] if ft_gens else None),
                "category": ex.get("category"),
            }
            for i, (ex, b) in enumerate(zip(examples[:30], base_gens[:30]))
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n[eval-llm] full results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
