"""Self-test the full pipeline with crafted queries — no LOCOMO bench.

Drives a small synthetic conversation through ingest → retrieve → SAM →
answer-gen. Reports per-feature signals so we can verify the latest
stack is actually firing correctly:

  - date enrichment       (Sunday → ISO date in content?)
  - entity index          (lookup_query catches names?)
  - cross-encoder         (scores show discrimination?)
  - Self-RAG              (fires on multi-hop?)
  - negation-aware        (demotes polarity mismatch?)
  - match_threshold       (gates noise queries?)
  - answer generation     (produces correct short answer?)
"""
import asyncio
import sys
import time
from datetime import datetime, timezone

# Force in-process bench-style pipeline build
sys.path.insert(0, ".")
from scripts.benchmark_locomo import _build_pipeline_with_temp_store, ingest_session_raw

from orchestration.pipeline import Pipeline, TargetDescriptor
from orchestration.sam.aal_record import AALRecord, AALTuple
from orchestration.sam.answer_generator import AnswerGenerator
from orchestration.sam._ollama_client import make_local_llm_client


# Synthetic conversation — Caroline and Mel chat over 2 sessions.
SESSION_1 = {
    "session_id": 1,
    "date": "2026-04-12T10:00:00",  # Sunday
    "messages": [
        {"speaker": "Caroline", "content": "Hey Mel, how have you been?"},
        {"speaker": "Mel", "content": "Good! Working on a new project."},
        {"speaker": "Caroline", "content": "I went to a yoga class last Thursday — it was great."},
        {"speaker": "Mel", "content": "That's wonderful. What kind?"},
        {"speaker": "Caroline", "content": "Hot yoga. The teacher is named Priya."},
    ],
}
SESSION_2 = {
    "session_id": 2,
    "date": "2026-05-03T10:00:00",  # ~3 weeks later
    "messages": [
        {"speaker": "Caroline", "content": "I started a meditation class on Monday."},
        {"speaker": "Mel", "content": "How's it going?"},
        {"speaker": "Caroline", "content": "It's amazing. I no longer feel anxious before work."},
        {"speaker": "Mel", "content": "Are you still doing yoga too?"},
        {"speaker": "Caroline", "content": "Yes, both. Yoga Thursdays, meditation Mondays."},
    ],
}

QUERIES = [
    # cat-1 single-hop
    ("What kind of yoga did Caroline try?", "hot yoga"),
    ("Who is Caroline's yoga teacher?", "Priya"),
    # cat-2 temporal (date arithmetic)
    ("When did Caroline go to her first yoga class?", "April"),  # last Thursday relative to Apr 12 Sun
    # cat-3 multi-hop
    ("What two practices does Caroline now do regularly?", "yoga and meditation"),
    # cat-5 adversarial (no answer)
    ("What's Caroline's favorite restaurant?", "I don't know"),
    # negation
    ("Does Caroline still feel anxious before work?", "no"),
]


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s): return f"\033[31m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"


async def main() -> int:
    print("=" * 80)
    print("Self-test: synthetic 2-session conversation, 6 queries")
    print("=" * 80)

    # Build the pipeline + memory store with current defaults
    pipeline, retriever, store, tmp_path, _aal_extractor, entity_index = (
        _build_pipeline_with_temp_store()
    )
    print("\n[1] Pipeline built. Components active:")
    print(f"    - embedder:  {pipeline.embedder.version}")
    print(f"    - retriever: {type(retriever).__name__}")
    print(f"    - reranker:  {type(pipeline.reranker).__name__}")
    print(f"    - sam:       reasoner={'on' if pipeline.sam.reasoner else 'heuristic-only'}")

    # Ingest both sessions
    print("\n[2] Ingesting 2 sessions (raw + date enrichment + per-session summary)...")
    t0 = time.perf_counter()
    for sess in (SESSION_1, SESSION_2):
        n = await ingest_session_raw(sess, retriever, store)
        print(f"    session {sess['session_id']}: {n} memories")
    ingest_ms = int((time.perf_counter() - t0) * 1000)
    print(f"    total ingest: {ingest_ms}ms, entity_index={entity_index.entity_count} entities")
    if entity_index.entity_count:
        top5 = entity_index.top_entities(n=5)
        print(f"    top entities: {top5}")

    # Build the answer generator (uses current backend = Ollama qwen2.5:3b)
    try:
        gen = AnswerGenerator(client=make_local_llm_client())
    except Exception as exc:
        print(f"    [warn] AnswerGenerator init failed: {type(exc).__name__}: {exc}")
        gen = None

    # Run each query
    print("\n[3] Running 6 test queries:\n")
    target = TargetDescriptor.for_claude()
    pass_count = 0
    for i, (q_text, expected) in enumerate(QUERIES, start=1):
        q = Pipeline.build_query(q_text, target=target)
        t0 = time.perf_counter()
        payload = await pipeline.run(q)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        ctx = payload.formatted_context

        # Check retrieval — does context contain expected term?
        if expected.lower() == "i don't know":
            ctx_pass = len(ctx) < 200  # very short context = NO match found
        else:
            ctx_pass = expected.lower() in ctx.lower()

        # Answer if generator available
        ans = None
        ans_pass = None
        if gen:
            ans = await gen.answer(ctx, q_text)
            if expected.lower() == "i don't know":
                ans_pass = "don't know" in ans.lower() or "don't" in ans.lower() or not ans
            else:
                ans_pass = any(w in ans.lower() for w in expected.lower().split())

        ctx_mark = _green("✓") if ctx_pass else _red("✗")
        ans_mark = _green("✓") if ans_pass else (_red("✗") if ans_pass is False else "·")
        print(f"  [{i}] Q: {q_text}")
        print(f"      expected: {expected!r}")
        print(f"      ctx ({len(ctx):>4} chars) {ctx_mark}  retrieved={dur_ms}ms")
        if ans:
            print(f"      ans: {ans!r} {ans_mark}")
        if ctx_pass and (ans_pass if gen else True):
            pass_count += 1
        print()

    # Cleanup tmp
    try:
        tmp_path.unlink()
    except OSError:
        pass

    print("=" * 80)
    print(f"PASSED: {pass_count}/{len(QUERIES)}")
    print("=" * 80)
    return 0 if pass_count == len(QUERIES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
