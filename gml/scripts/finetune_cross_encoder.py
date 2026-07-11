"""Fine-tune a cross-encoder on LOCOMO QA triples.

Loads (query, positive, negative) triples produced by prepare_ft_data.py
and converts each into TWO binary-classification examples:
    (query, positive) → 1.0
    (query, negative) → 0.0

Then fine-tunes ``BAAI/bge-reranker-base`` with the standard
CrossEncoder.fit() loop. Uses Apple MPS when available, else CPU.

Output: a fine-tuned model directory at ``--out`` that can be loaded
back by ``SentenceTransformerCrossEncoder(model_name=<path>)``.

Run:
    .venv/bin/python scripts/finetune_cross_encoder.py \\
        --data-dir /tmp/locomo-ft-data \\
        --out      models/ce_locomo_ft \\
        --base     BAAI/bge-reranker-base \\
        --epochs   1 \\
        --batch    8
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _triples_to_examples(triples: list[dict]) -> list:
    """Each triple → 2 binary examples (positive, negative)."""
    from sentence_transformers import InputExample
    out = []
    for t in triples:
        q = t.get("query", "")
        p = t.get("positive", "")
        n = t.get("negative", "")
        if not q or not p or not n:
            continue
        out.append(InputExample(texts=[q, p], label=1.0))
        out.append(InputExample(texts=[q, n], label=0.0))
    return out


def _eval_on_test(model, test_triples: list[dict]) -> dict:
    """Cheap eval: average margin between positive and negative scores
    across triples (and average ranking accuracy: how often positive > negative).
    """
    if not test_triples:
        return {}
    pos_pairs = [(t["query"], t["positive"]) for t in test_triples]
    neg_pairs = [(t["query"], t["negative"]) for t in test_triples]
    p_scores = model.predict(pos_pairs, batch_size=32, show_progress_bar=False)
    n_scores = model.predict(neg_pairs, batch_size=32, show_progress_bar=False)
    margins = [float(p) - float(n) for p, n in zip(p_scores, n_scores)]
    acc = sum(1 for m in margins if m > 0) / len(margins)
    return {
        "n_triples": len(test_triples),
        "ranking_accuracy": acc,
        "avg_margin": sum(margins) / len(margins),
        "avg_pos_score": sum(float(p) for p in p_scores) / len(p_scores),
        "avg_neg_score": sum(float(n) for n in n_scores) / len(n_scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default="BAAI/bge-reranker-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--device", default=None,
                        help="Override torch device (mps / cuda / cpu)")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Cap training triples (0 = all). Useful for smoke tests.")
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader
    import torch

    # Pick device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[ft-ce] device: {device}")

    # Load data
    data_dir = Path(args.data_dir)
    train_triples = _load_jsonl(data_dir / "train.jsonl")
    test_triples = _load_jsonl(data_dir / "test.jsonl")
    if args.max_train and args.max_train < len(train_triples):
        import random
        random.Random(42).shuffle(train_triples)
        train_triples = train_triples[: args.max_train]
    print(f"[ft-ce] loaded {len(train_triples)} train / {len(test_triples)} test triples")

    train_examples = _triples_to_examples(train_triples)
    print(f"[ft-ce] {len(train_examples)} binary training examples (2× triples)")

    # Build model — load base + move to device
    print(f"[ft-ce] loading base model: {args.base}")
    model = CrossEncoder(args.base, device=device)

    # ---- Pre-train eval -----------------------------------------------
    print("[ft-ce] pre-train evaluation on test set...")
    pre = _eval_on_test(model, test_triples)
    print(f"[ft-ce]   pre-train: ranking_acc={pre.get('ranking_accuracy', 0):.3f}  "
          f"avg_margin={pre.get('avg_margin', 0):.3f}  "
          f"pos={pre.get('avg_pos_score', 0):.3f}  neg={pre.get('avg_neg_score', 0):.3f}")

    # ---- Train --------------------------------------------------------
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch)
    warmup = int(len(train_loader) * args.epochs * args.warmup_frac)
    print(f"[ft-ce] training: epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} warmup={warmup}")

    t0 = time.time()
    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"[ft-ce] training done in {elapsed/60:.1f} min")

    # ---- Save ---------------------------------------------------------
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[ft-ce] saved fine-tuned model to {out_path}")

    # ---- Post-train eval ----------------------------------------------
    print("[ft-ce] post-train evaluation on test set...")
    post = _eval_on_test(model, test_triples)
    print(f"[ft-ce]   post-train: ranking_acc={post.get('ranking_accuracy', 0):.3f}  "
          f"avg_margin={post.get('avg_margin', 0):.3f}  "
          f"pos={post.get('avg_pos_score', 0):.3f}  neg={post.get('avg_neg_score', 0):.3f}")

    # ---- Summary ------------------------------------------------------
    summary = {
        "base_model": args.base,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "train_triples": len(train_triples),
        "test_triples": len(test_triples),
        "training_minutes": elapsed / 60,
        "pre_eval": pre,
        "post_eval": post,
        "out_path": str(out_path),
    }
    (out_path / "ft_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ft-ce] summary: {out_path / 'ft_summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
