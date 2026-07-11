"""Hard-negative mining for cross-encoder fine-tuning round 2.

The first FT pass (FT-1) used easy + random negatives. Now that we have
a stronger CE (Recall@5 = 0.70), we can use IT to find candidates that
look-correct-but-aren't — the "hard negatives" the model still confuses.

Algorithm:
  For each (query, gold_evidence) in the train set:
    1. Take all messages from the same conversation (minus the gold)
    2. Score them with the current FT'd CE
    3. Top-K scoring NON-gold messages = hard negatives
    4. Emit (query, gold, hard_negative) triples — one per hard negative

These hard negatives have high CE scores (model thinks they're relevant)
but are actually wrong — forcing the next training pass to learn finer
distinctions. Standard recipe; typically lifts R@5 by 0.08-0.12.

Input:  /tmp/locomo-ft-data/train.jsonl  (original triples)
Output: /tmp/locomo-ft-data-hard/train.jsonl  (mined triples)
        — same shape as the original so finetune_cross_encoder.py
          works without changes.

Run:
    .venv/bin/python scripts/mine_hard_negatives.py \\
        --ce-model models/ce_locomo_ft \\
        --data /tmp/locomo/locomo10.json \\
        --in-data /tmp/locomo-ft-data \\
        --out-dir /tmp/locomo-ft-data-hard \\
        --negs-per-positive 4 \\
        --candidates-per-query 50
"""
import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


def _index_conversation(conv: dict) -> dict[str, str]:
    """dia_id → speaker-prefixed message text (same as prepare_ft_data)."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ce-model", required=True, help="Path to FT'd cross-encoder")
    p.add_argument("--data", required=True, help="Original LOCOMO JSON")
    p.add_argument("--in-data", required=True, help="Original FT data dir (with train.jsonl)")
    p.add_argument("--out-dir", required=True, help="Where to write mined triples")
    p.add_argument("--negs-per-positive", type=int, default=4)
    p.add_argument("--candidates-per-query", type=int, default=50,
                   help="How many same-conv messages to score per query")
    p.add_argument("--test-conv-indices", default="8,9",
                   help="Held-out convs to skip (must match prepare_ft_data)")
    p.add_argument("--max-queries", type=int, default=0,
                   help="Cap mining for smoke tests")
    args = p.parse_args()

    from sentence_transformers import CrossEncoder
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[mine-hn] device: {device}")
    print(f"[mine-hn] loading CE: {args.ce_model}")
    ce = CrossEncoder(args.ce_model, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LOCOMO + build per-conversation indices
    raw = json.loads(Path(args.data).read_text())
    test_set = set(int(x) for x in args.test_conv_indices.split(","))
    conv_indices = [_index_conversation(c) for c in raw]
    print(f"[mine-hn] loaded {len(raw)} conversations")

    # Gather all (query, gold_evidence_ids, conv_idx) train tuples
    # — group by query+conv so we only score each query's candidates once.
    train_jobs: list[dict] = []
    seen_queries: set[tuple[int, str]] = set()
    for ci, conv in enumerate(raw):
        if ci in test_set:
            continue
        idx = conv_indices[ci]
        for qa in conv.get("qa", []):
            q = (qa.get("question") or "").strip()
            evidence = qa.get("evidence") or []
            if not q or not isinstance(evidence, list) or not evidence:
                continue
            gold_ids = {e for e in evidence if isinstance(e, str) and e in idx}
            if not gold_ids:
                continue
            key = (ci, q)
            if key in seen_queries:
                continue
            seen_queries.add(key)
            train_jobs.append({
                "conv_idx": ci, "query": q, "gold_ids": gold_ids,
                "category": qa.get("category"),
            })

    if args.max_queries:
        train_jobs = train_jobs[: args.max_queries]
    print(f"[mine-hn] {len(train_jobs)} unique train queries to mine")

    # ---- Mine -------------------------------------------------------
    out_path = out_dir / "train.jsonl"
    t0 = time.time()
    n_triples_written = 0
    pbar_every = max(1, len(train_jobs) // 20)

    with out_path.open("w") as f:
        for ji, job in enumerate(train_jobs):
            conv_msgs = conv_indices[job["conv_idx"]]
            candidates = [
                (did, text) for did, text in conv_msgs.items()
                if did not in job["gold_ids"]
            ]
            if not candidates:
                continue

            # Score in batches (we may have hundreds of candidates)
            pairs = [[job["query"], text] for _, text in candidates]
            scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)

            # Pick top-K highest-scoring candidates (= hardest negatives)
            scored = sorted(
                zip(candidates, scores),
                key=lambda x: -float(x[1]),
            )[: args.candidates_per_query]
            # Out of the top-N, take the first args.negs_per_positive as negs
            hard_negs = [text for (_, text), _ in scored[: args.negs_per_positive]]

            # Emit one triple per (positive, hard_neg) pair
            for gold_id in job["gold_ids"]:
                positive_text = conv_msgs.get(gold_id, "")
                if not positive_text:
                    continue
                for neg in hard_negs:
                    if neg == positive_text:
                        continue
                    f.write(json.dumps({
                        "query": job["query"],
                        "positive": positive_text,
                        "negative": neg,
                        "category": job["category"],
                        "mining": "hard",
                    }) + "\n")
                    n_triples_written += 1

            if (ji + 1) % pbar_every == 0:
                elapsed = time.time() - t0
                rate = (ji + 1) / max(elapsed, 1e-9)
                eta = (len(train_jobs) - ji - 1) / max(rate, 1e-9)
                print(f"  {ji+1}/{len(train_jobs)} queries · "
                      f"{n_triples_written} triples · "
                      f"{rate:.1f} q/s · ETA {eta/60:.1f}m",
                      flush=True)

    elapsed = time.time() - t0
    print(f"[mine-hn] wrote {n_triples_written} hard-negative triples to {out_path}")
    print(f"[mine-hn] mining took {elapsed/60:.1f} min")

    # Also copy the original test.jsonl into the new dir so the trainer
    # can compare pre/post on the same held-out set.
    src_test = Path(args.in_data) / "test.jsonl"
    if src_test.exists():
        dst_test = out_dir / "test.jsonl"
        dst_test.write_text(src_test.read_text())
        print(f"[mine-hn] copied original test.jsonl ({dst_test})")

    stats = {
        "ce_model": args.ce_model,
        "queries_mined": len(train_jobs),
        "triples_written": n_triples_written,
        "negs_per_positive": args.negs_per_positive,
        "candidates_per_query": args.candidates_per_query,
        "mining_minutes": round(elapsed / 60, 1),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[mine-hn] summary: {out_dir / 'stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
