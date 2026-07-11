"""Run one query through the full GML pipeline and print every stage's output.

Seeds three memories with a deliberate supersession (Stripe → Adyen) so that
conflict resolution actually fires, then asks "what payment provider do we use?"
and prints what Classifier → Embedder → Retriever → Reranker → SAM → Assembler →
Translator each saw and produced.

Run:
    .venv/bin/python scripts/demo_pipeline_trace.py
"""
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Use vanilla qwen2.5:3b (small, fast) rather than the FT'd path so this
# demo runs in ~10s instead of 60s. The FT'd path is the same code path.
os.environ.setdefault("GML_LLM_BACKEND", "ollama")
os.environ.setdefault("GML_OLLAMA_MODEL", "qwen2.5:3b")
# Force SAM on so conflict-resolution actually runs (default would skip
# because the question may classify as simple single-hop).
os.environ["GML_FORCE_SAM"] = "1"

# Sandbox the memory store into a temp dir so this demo doesn't touch
# ~/.gml/memories.jsonl.
_tmpdir = Path(tempfile.mkdtemp(prefix="gml-demo-"))
_mem_path = _tmpdir / "memories.jsonl"
os.environ["GML_MEMORY_STORE_PATH"] = str(_mem_path)


async def main() -> None:
    from orchestration.pipeline.contracts import MemoryItem
    from orchestration.server import build_default_state, stream_pipeline_trace
    from orchestration.embedder import FastEmbedEmbedder

    print("► Building pipeline state (loading models)…\n")
    embedder = FastEmbedEmbedder()
    state = await build_default_state(
        embedder=embedder,
        memory_path=_mem_path,
        default_target_name="claude",
    )

    # Seed three memories. Two contradict; the most recent should win.
    now = datetime.now(timezone.utc)
    seed = [
        MemoryItem(
            id="m-stripe",
            content="We use Stripe for payments.",
            entity="payments", attribute="provider", value="Stripe",
            source="conversation",
            authority_score=0.7,
            timestamp=now - timedelta(days=90),
        ),
        MemoryItem(
            id="m-adyen",
            content="We switched payment processing from Stripe to Adyen last quarter.",
            entity="payments", attribute="provider", value="Adyen",
            source="conversation",
            authority_score=0.7,
            timestamp=now - timedelta(days=7),
        ),
        MemoryItem(
            id="m-postgres",
            content="Our orders database runs on PostgreSQL 16.",
            entity="orders_db", attribute="engine", value="PostgreSQL 16",
            source="conversation",
            authority_score=0.7,
            timestamp=now - timedelta(days=30),
        ),
    ]
    await state.memory_store.add_many(seed)
    await state.retriever.ingest(seed)
    print(f"  seeded {len(seed)} memories ({_mem_path})\n")

    question = "What payment provider do we use?"
    print(f"► Query: {question!r}\n")
    print("=" * 78)

    final = None
    async for kind, payload in stream_pipeline_trace(state, question):
        if kind == "stage":
            _print_stage(payload)
        elif kind == "done":
            final = payload

    print("=" * 78)
    if final:
        print("\n► FINAL formatted_context (what the target AI would receive):\n")
        ctx = final.get("formatted_context", "")
        for line in ctx.splitlines():
            print(f"    {line}")

        ann = final.get("annotations", {}) or {}
        if ann.get("sam_reasoning"):
            print(f"\n► SAM reasoning (LLM's explanation for the drops):\n")
            for line in (ann["sam_reasoning"] or "").splitlines():
                print(f"    {line}")


def _print_stage(stage: dict) -> None:
    name = stage["stage"]
    impl = stage.get("name", "")
    dur = stage["duration_ms"]
    print(f"\n[{name}]  ({impl})  {dur}ms")
    out = stage.get("output", {})
    body = json.dumps(out, indent=2, default=str)
    # Truncate any field longer than 200 chars to keep output readable.
    for line in body.splitlines():
        if len(line) > 200:
            line = line[:197] + "..."
        print(f"    {line}")


if __name__ == "__main__":
    asyncio.run(main())
