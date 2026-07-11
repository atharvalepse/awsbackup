"""Prepare answer-LLM fine-tuning data from LOCOMO.

Output: JSONL of {prompt, completion} where:
  prompt     = "Context: <evidence...>\nQuestion: <q>\nAnswer:"
  completion = "<gold_answer>"

This is straightforward supervised instruction-tuning data. The same
8-train / 2-test conversation split as prepare_ft_data.py keeps the
holdout consistent across all three fine-tunes.

For each LOCOMO QA:
  - Resolve evidence dia_ids → concatenated evidence text
  - Use ``answer`` if present, else ``adversarial_answer``
  - Wrap in our answer prompt format

Run:
    .venv/bin/python scripts/prepare_ft_llm_data.py \\
        --data /tmp/locomo/locomo10.json \\
        --out-dir /tmp/locomo-llm-ft-data
"""
import argparse
import json
import re
import sys
from pathlib import Path


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
    # Gold answer
    gold = qa.get("answer") or qa.get("adversarial_answer") or ""
    if not gold:
        return None
    # Resolve evidence
    evidence_ids = qa.get("evidence") or []
    evidence_texts: list[str] = []
    for eid in evidence_ids:
        if isinstance(eid, str) and eid in conv_idx:
            evidence_texts.append(conv_idx[eid])
    if not evidence_texts:
        return None
    context = "\n".join(f"<memory>{t}</memory>" for t in evidence_texts)
    prompt = (
        f"Below is some context from a long-running conversation, followed by "
        f"a question. Answer the question concisely.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {q}\n\n"
        f"ANSWER:"
    )
    return {
        "prompt": prompt,
        "completion": str(gold),
        "category": qa.get("category"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--test-conv-indices", default="8,9")
    args = p.parse_args()

    raw = json.loads(Path(args.data).read_text())
    if not isinstance(raw, list):
        print(f"[err] expected LOCOMO list", file=sys.stderr)
        return 1

    test_idx_set = set(int(x) for x in args.test_conv_indices.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ex: list[dict] = []
    test_ex: list[dict] = []
    cat_counts: dict[int, int] = {}

    for ci, conv in enumerate(raw):
        idx = _index_conversation(conv)
        target = test_ex if ci in test_idx_set else train_ex
        for qa in conv.get("qa", []):
            ex = _build_example(qa, idx)
            if ex is None:
                continue
            target.append(ex)
            cat = ex.get("category") or 0
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    def _write(path: Path, items: list[dict]) -> None:
        with path.open("w") as f:
            for ex in items:
                f.write(json.dumps(ex) + "\n")

    _write(out_dir / "train.jsonl", train_ex)
    _write(out_dir / "test.jsonl", test_ex)

    stats = {
        "train_examples": len(train_ex),
        "test_examples": len(test_ex),
        "test_conv_indices": sorted(test_idx_set),
        "categories": cat_counts,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"wrote {len(train_ex)} train / {len(test_ex)} test examples")
    print(f"categories: {cat_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
