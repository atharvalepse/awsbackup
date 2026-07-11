"""Targeted retrieval-only eval: Recall@K on LOCOMO test conversations.

NOT a full LOCOMO bench. This script ONLY measures the retrieval stage
(embedder + retriever + cross-encoder) — not SAM, not answer-gen. The
metric is straightforward:

    For each QA with evidence pointer (e.g. "D2:8"), check whether the
    evidence message is in the cross-encoder's top-K output.

Use this to attribute changes — does the fine-tuned cross-encoder
actually lift Recall@K, or just sharpen margins?

Compares two configurations side-by-side: a baseline and a candidate
(typically the fine-tuned model).

Run:
    .venv/bin/python scripts/eval_retrieval.py \\
        --data /tmp/locomo/locomo10.json \\
        --test-conv-indices 8,9 \\
        --base-ce BAAI/bge-reranker-base \\
        --cand-ce models/ce_locomo_ft \\
        --k-values 5,10,20
"""
import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

# Import bench helpers
sys.path.insert(0, ".")


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


def _normalize_locomo_native(raw):
    """Reuse the same loader as the bench."""
    if isinstance(raw, dict) and "conversations" in raw:
        return raw
    if isinstance(raw, list):
        convs = raw
    else:
        convs = [raw]
    out_convs = []
    for conv in convs:
        conv_id = conv.get("sample_id") or conv.get("id") or f"conv-{len(out_convs)+1}"
        block = conv.get("conversation", conv)
        sessions = []
        session_keys = sorted(
            (k for k in block if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        for sk in session_keys:
            n = int(sk.split("_")[1])
            messages = []
            for m in block[sk]:
                content = m.get("text") or m.get("content") or ""
                messages.append({
                    "speaker": m.get("speaker") or "speaker",
                    "content": content,
                    "dia_id": m.get("dia_id"),
                })
            sessions.append({
                "session_id": n,
                "date": block.get(f"{sk}_date_time"),
                "messages": messages,
            })
        out_convs.append({
            "id": conv_id, "sessions": sessions,
            "qa": conv.get("qa", []),
            # Keep the raw conversation block so _index_conversation works.
            "_raw_conv_block": block,
        })
    return {"conversations": out_convs}


async def _ingest_conv_into_temp_pipeline(conv, ce_model: str | None) -> tuple:
    """Build a fresh pipeline + ingest one conversation's messages.

    Returns (pipeline, retriever, dia_id_to_memory_id) so we can later
    score retrieval by checking if gold dia_id's MemoryItem.id appears in
    the top-K output.

    ``ce_model`` controls which cross-encoder is wired into the pipeline.
    Pass None to use whatever GML_CE_BACKEND / GML_ST_CE_MODEL env say.
    """
    import os
    if ce_model:
        os.environ["GML_CE_BACKEND"] = "st"
        os.environ["GML_ST_CE_MODEL"] = ce_model

    from scripts.benchmark_locomo import _build_pipeline_with_temp_store
    from datetime import datetime, timezone
    from orchestration.pipeline.contracts import MemoryItem
    import uuid

    pipeline, retriever, store, tmp_path, _aal, _idx = _build_pipeline_with_temp_store()

    # Ingest each message as its own MemoryItem; track dia_id → memory_id
    dia_to_id: dict[str, str] = {}
    block = conv.get("_raw_conv_block", {})
    items: list[MemoryItem] = []

    sess_date_keys = {k for k in block if k.endswith("_date_time")}
    for key, messages in block.items():
        if not key.startswith("session_") or key in sess_date_keys:
            continue
        if not isinstance(messages, list):
            continue
        sess_n = int(key.split("_")[1])
        sess_date_str = block.get(f"{key}_date_time")
        try:
            ts = (datetime.fromisoformat(sess_date_str.replace("Z", "+00:00"))
                  if sess_date_str else datetime.now(timezone.utc))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        for m in messages:
            did = m.get("dia_id")
            text = m.get("text") or m.get("content") or ""
            speaker = m.get("speaker") or "speaker"
            if not did or not text:
                continue
            mem_id = f"eval-{uuid.uuid4().hex[:12]}"
            dia_to_id[did] = mem_id
            items.append(MemoryItem(
                id=mem_id,
                content=f"{speaker}: {text}",
                summary_short=text[:120],
                entity=speaker, attribute=None, value=None,
                timestamp=ts,
                source="locomo-eval",
                authority_score=0.7, pinned=False,
                raw_metadata={"session": sess_n, "dia_id": did},
            ))
    store.add_many(items)
    await retriever.ingest(items)
    return pipeline, retriever, dia_to_id, tmp_path


async def _run_eval(
    data_path: Path, test_indices: list[int], ce_model: str | None,
    k_values: list[int], max_qa_per_conv: int = 0,
) -> dict:
    raw = json.loads(data_path.read_text())
    norm = _normalize_locomo_native(raw)
    convs = norm["conversations"]
    selected = [convs[i] for i in test_indices if i < len(convs)]

    from orchestration.pipeline import Pipeline
    from orchestration.pipeline.contracts import TargetDescriptor

    results = {f"recall_at_{k}": [] for k in k_values}
    per_cat = {f"recall_at_{k}": {} for k in k_values}
    n_total = 0

    for conv in selected:
        print(f"  conv {conv['id']}: ingesting {sum(len(s.get('messages', [])) for s in conv.get('sessions', []))} messages...", flush=True)
        t0 = time.perf_counter()
        pipeline, retriever, dia_to_id, tmp_path = await _ingest_conv_into_temp_pipeline(
            conv, ce_model=ce_model,
        )
        ingest_ms = int((time.perf_counter() - t0) * 1000)
        target = TargetDescriptor.for_claude()
        qas = conv.get("qa", [])
        if max_qa_per_conv and max_qa_per_conv < len(qas):
            qas = qas[:max_qa_per_conv]
        print(f"    ingested in {ingest_ms}ms; eval {len(qas)} QAs", flush=True)

        for qa in qas:
            q = qa.get("question", "").strip()
            evidence = qa.get("evidence") or []
            cat = qa.get("category")
            if not q or not isinstance(evidence, list) or not evidence:
                continue
            gold_mem_ids = {dia_to_id[eid] for eid in evidence if eid in dia_to_id}
            if not gold_mem_ids:
                continue
            # Build query, embed + retrieve + rerank (do NOT call SAM/answer)
            qobj = Pipeline.build_query(q, target=target)
            embedded = await pipeline.embedder.embed(
                qobj, await pipeline.classifier.classify(qobj),
            )
            # Use top-100 candidates for cross-encoder
            top_n = max(k_values) * 4
            hits = await pipeline.retriever.get_top_matches(embedded, k=top_n)
            if not hits:
                # Record zero hits
                for k in k_values:
                    results[f"recall_at_{k}"].append(0)
                    per_cat[f"recall_at_{k}"].setdefault(cat, []).append(0)
                continue
            ranked = await pipeline.reranker.pick_best(hits, qobj, k=max(k_values))
            top_ids = [r.hit.record.id for r in ranked]
            n_total += 1
            for k in k_values:
                hit = 1 if any(rid in gold_mem_ids for rid in top_ids[:k]) else 0
                results[f"recall_at_{k}"].append(hit)
                per_cat[f"recall_at_{k}"].setdefault(cat, []).append(hit)

        try:
            tmp_path.unlink()
        except OSError:
            pass

    summary = {"n": n_total}
    for k in k_values:
        scores = results[f"recall_at_{k}"]
        summary[f"recall_at_{k}"] = sum(scores) / max(len(scores), 1)
        summary[f"recall_at_{k}_per_cat"] = {
            str(c): sum(v) / max(len(v), 1)
            for c, v in per_cat[f"recall_at_{k}"].items()
        }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--test-conv-indices", default="8,9")
    p.add_argument("--base-ce", default="BAAI/bge-reranker-base")
    p.add_argument("--cand-ce", required=True,
                   help="Candidate CE model (path or HF id) to compare against base")
    p.add_argument("--k-values", default="5,10,20")
    p.add_argument("--max-qa-per-conv", type=int, default=30)
    args = p.parse_args()

    test_indices = [int(x) for x in args.test_conv_indices.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]

    print(f"\n=== BASELINE: {args.base_ce} ===")
    base = asyncio.run(_run_eval(
        Path(args.data), test_indices, args.base_ce, k_values,
        max_qa_per_conv=args.max_qa_per_conv,
    ))

    print(f"\n=== CANDIDATE: {args.cand_ce} ===")
    cand = asyncio.run(_run_eval(
        Path(args.data), test_indices, args.cand_ce, k_values,
        max_qa_per_conv=args.max_qa_per_conv,
    ))

    print("\n" + "=" * 72)
    print(f"RETRIEVAL EVAL  (n={base['n']} QAs, {len(test_indices)} test convs)")
    print("=" * 72)
    print(f"{'metric':<22} {'baseline':>12} {'candidate':>12} {'Δ':>10}")
    print("-" * 72)
    for k in k_values:
        b = base[f"recall_at_{k}"]
        c = cand[f"recall_at_{k}"]
        delta = c - b
        sign = "+" if delta >= 0 else ""
        print(f"  Recall@{k:<14}    {b:>12.3f} {c:>12.3f} {sign}{delta:>9.3f}")

    print("\n--- per-category Recall@10 ---")
    cats = sorted(set(base["recall_at_10_per_cat"].keys()) | set(cand["recall_at_10_per_cat"].keys()))
    for cat in cats:
        b = base["recall_at_10_per_cat"].get(cat, 0)
        c = cand["recall_at_10_per_cat"].get(cat, 0)
        print(f"  cat-{cat:<3}              {b:>12.3f} {c:>12.3f} {'+' if c >= b else ''}{c-b:>9.3f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
