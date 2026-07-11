"""Prepare cross-encoder fine-tuning data from LOCOMO.

Output: JSONL files with (query, positive_doc, negative_doc) triples.

For each LOCOMO QA with an evidence pointer (e.g. "D2:8" — session 2,
dialogue 8), the evidence message text becomes the POSITIVE example.
Negatives are sampled in two ways:
  - HARD: other messages from the SAME conversation (semantically close)
  - EASY: messages from OTHER conversations (clearly unrelated)

By default we emit 4 negatives per positive: 2 hard, 2 easy. The split
is 8 conversations for train, 2 for test.

Run:
    .venv/bin/python scripts/prepare_ft_data.py \\
        --data /tmp/locomo/locomo10.json \\
        --out-dir /tmp/locomo-ft-data \\
        --negatives-per-positive 4

Outputs:
    {out-dir}/train.jsonl   — ~train_n × (1+negs) lines
    {out-dir}/test.jsonl    — held-out 2 convs
    {out-dir}/stats.json    — counts, split info
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


_DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")


def _index_conversation(conv: dict) -> dict[str, str]:
    """Map dia_id → message text for fast lookup."""
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
            dia_id = m.get("dia_id")
            text = m.get("text") or m.get("content") or ""
            if dia_id and text:
                # Prepend speaker for context-aware embedding
                speaker = m.get("speaker") or ""
                text_with_speaker = f"{speaker}: {text}" if speaker else text
                out[dia_id] = text_with_speaker
    return out


def _all_message_texts(conv_index: dict[str, str]) -> list[tuple[str, str]]:
    """Return [(dia_id, text), ...] sorted by dia_id."""
    return sorted(conv_index.items(), key=lambda kv: kv[0])


def _emit_triples(
    qa: dict,
    conv_index: dict[str, str],
    other_convs_messages: list[str],
    negatives_per_positive: int,
    rng: random.Random,
) -> list[dict]:
    """For ONE QA, emit (query, positive, negative) triples.

    Returns 0 triples if the QA has no usable evidence.
    """
    question = qa.get("question", "").strip()
    if not question:
        return []
    # LOCOMO cat-5 carries `adversarial_answer` not `answer`; both are
    # valid for our purposes (the question still has evidence).
    evidence = qa.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        return []

    # Resolve evidence dia_ids → message texts (positives)
    positives: list[str] = []
    evidence_ids: set[str] = set()
    for e in evidence:
        if not isinstance(e, str):
            continue
        # Sometimes evidence has trailing ":n" patterns we keep verbatim
        if e in conv_index:
            evidence_ids.add(e)
            positives.append(conv_index[e])

    if not positives:
        return []

    # Hard negatives: same conversation, but NOT the evidence messages
    same_conv_msgs = [
        (did, text) for did, text in conv_index.items()
        if did not in evidence_ids
    ]
    if not same_conv_msgs:
        return []

    # Easy negatives: random messages from other conversations
    triples: list[dict] = []
    for positive in positives:
        n_hard = max(1, negatives_per_positive // 2)
        n_easy = negatives_per_positive - n_hard
        hard_picks = rng.sample(
            same_conv_msgs, k=min(n_hard, len(same_conv_msgs))
        )
        easy_picks = (
            rng.sample(other_convs_messages, k=min(n_easy, len(other_convs_messages)))
            if other_convs_messages else []
        )
        negatives = [t for _, t in hard_picks] + easy_picks

        for negative in negatives:
            triples.append({
                "query": question,
                "positive": positive,
                "negative": negative,
                "category": qa.get("category"),
            })
    return triples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to LOCOMO JSON")
    p.add_argument("--out-dir", required=True, help="Output dir for train/test JSONL")
    p.add_argument("--negatives-per-positive", type=int, default=4)
    p.add_argument("--test-conv-indices", type=str, default="8,9",
                   help="Conversation indices to hold out (comma-separated)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(args.data).read_text())
    if not isinstance(raw, list):
        print(f"[err] expected LOCOMO list, got {type(raw).__name__}", file=sys.stderr)
        return 1

    # Index every conversation's messages
    conv_indices: list[dict[str, str]] = [_index_conversation(c) for c in raw]
    print(f"loaded {len(raw)} conversations; "
          f"avg {sum(len(idx) for idx in conv_indices) / max(len(conv_indices),1):.0f} messages each")

    test_idx_set = set(int(x.strip()) for x in args.test_conv_indices.split(",") if x.strip())
    print(f"test conv indices: {sorted(test_idx_set)}")

    train_triples: list[dict] = []
    test_triples: list[dict] = []
    cat_counts = defaultdict(int)

    for ci, conv in enumerate(raw):
        idx = conv_indices[ci]
        if not idx:
            continue
        # Build "other-convs" message pool for easy negatives
        other_msgs: list[str] = []
        for oi, other_idx in enumerate(conv_indices):
            if oi == ci:
                continue
            other_msgs.extend(other_idx.values())

        is_test = ci in test_idx_set
        target = test_triples if is_test else train_triples

        for qa in conv.get("qa", []):
            triples = _emit_triples(
                qa, idx, other_msgs, args.negatives_per_positive, rng,
            )
            target.extend(triples)
            if triples:
                cat_counts[qa.get("category", "?")] += 1

    def _write(path: Path, items: list[dict]) -> None:
        with path.open("w") as f:
            for t in items:
                f.write(json.dumps(t) + "\n")

    _write(out_dir / "train.jsonl", train_triples)
    _write(out_dir / "test.jsonl", test_triples)

    stats = {
        "train_triples": len(train_triples),
        "test_triples": len(test_triples),
        "test_conv_indices": sorted(test_idx_set),
        "categories_seen": dict(cat_counts),
        "negatives_per_positive": args.negatives_per_positive,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {out_dir / 'train.jsonl'}: {len(train_triples)} triples")
    print(f"wrote {out_dir / 'test.jsonl'}:  {len(test_triples)} triples")
    print(f"categories: {dict(cat_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
