"""Fine-tune bge-large-en-v1.5 on LOCOMO (query, evidence) pairs.

Uses sentence-transformers' MultipleNegativesRankingLoss: positives are
(question, evidence) pairs, negatives are other in-batch examples
(in-batch negative mining — no explicit negative sampling needed).

Same 8-train / 2-test conv split. Reuses the cross-encoder triples
file but only takes the (query, positive) pair from each triple (dedup
since multiple triples per query share the same positive).

Run:
    .venv/bin/python scripts/finetune_embedder.py \\
        --data-dir /tmp/locomo-ft-data \\
        --out models/embedder_locomo_ft \\
        --base BAAI/bge-large-en-v1.5 \\
        --epochs 1 --batch 8
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _load_pairs(path: Path) -> list[tuple[str, str]]:
    """Read a triples JSONL and return unique (query, positive) pairs."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            key = (t["query"], t["positive"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    import torch
    from sentence_transformers import (
        SentenceTransformer, InputExample, losses,
    )
    from torch.utils.data import DataLoader

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[ft-emb] device: {device}")

    data_dir = Path(args.data_dir)
    train_pairs = _load_pairs(data_dir / "train.jsonl")
    test_pairs = _load_pairs(data_dir / "test.jsonl")
    if args.max_pairs and args.max_pairs < len(train_pairs):
        train_pairs = train_pairs[: args.max_pairs]
    print(f"[ft-emb] {len(train_pairs)} train / {len(test_pairs)} test pairs (unique q+positive)")

    train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]

    # ---- Load base model --------------------------------------------
    print(f"[ft-emb] loading {args.base} on {device}")
    t0 = time.time()
    model = SentenceTransformer(args.base, device=device)
    print(f"[ft-emb] loaded in {time.time()-t0:.1f}s")

    # ---- Pre-train eval: cosine on a few test pairs -----------------
    print("[ft-emb] pre-train eval...")
    sample = test_pairs[:50]
    q_vecs = model.encode([q for q, _ in sample], convert_to_numpy=True, show_progress_bar=False)
    p_vecs = model.encode([p for _, p in sample], convert_to_numpy=True, show_progress_bar=False)
    import numpy as np
    sims = (q_vecs * p_vecs).sum(axis=1) / (
        np.linalg.norm(q_vecs, axis=1) * np.linalg.norm(p_vecs, axis=1) + 1e-9
    )
    pre_avg_sim = float(np.mean(sims))
    print(f"[ft-emb]   pre-train avg cosine sim (q, positive): {pre_avg_sim:.3f}")

    # ---- Train ------------------------------------------------------
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(train_loader) * args.epochs * 0.1)

    print(f"[ft-emb] training: epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} warmup={warmup} steps_per_epoch={len(train_loader)}")
    t0 = time.time()
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"[ft-emb] training done in {elapsed/60:.1f} min")

    # ---- Save -------------------------------------------------------
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[ft-emb] saved to {out_path}")

    # ---- Post-train eval --------------------------------------------
    q_vecs = model.encode([q for q, _ in sample], convert_to_numpy=True, show_progress_bar=False)
    p_vecs = model.encode([p for _, p in sample], convert_to_numpy=True, show_progress_bar=False)
    sims = (q_vecs * p_vecs).sum(axis=1) / (
        np.linalg.norm(q_vecs, axis=1) * np.linalg.norm(p_vecs, axis=1) + 1e-9
    )
    post_avg_sim = float(np.mean(sims))
    print(f"[ft-emb]   post-train avg cosine sim (q, positive): {post_avg_sim:.3f}  "
          f"(Δ {post_avg_sim - pre_avg_sim:+.3f})")

    summary = {
        "base_model": args.base,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "training_minutes": round(elapsed / 60, 1),
        "pre_avg_cos_sim": pre_avg_sim,
        "post_avg_cos_sim": post_avg_sim,
        "delta": post_avg_sim - pre_avg_sim,
        "out_path": str(out_path),
    }
    (out_path / "ft_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ft-emb] summary: {out_path / 'ft_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
