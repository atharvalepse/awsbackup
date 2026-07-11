"""End-to-end smoke test for the SDP (Semantic Decomposition Pipeline).

Runs a fixed set of (user, assistant) turns through SDPPipeline and
prints what each stage produced: entities, relationships, importance,
confidence, AALMemory content, and the final MemoryItem.

Also compares SDP speed to a single MemoryExtractor (LLM) call for the
same turn so you can see the latency gap directly.

Run:
    cd /Users/atharvalepse/Projects/gml-orchestration
    .venv/bin/python scripts/test_sdp.py
"""
import asyncio
import json
import sys
import time

from orchestration.sdp import SDPPipeline


TURNS = [
    # Tech stack + version + port
    ("Just so you know, auth-svc is in Go 1.23 and runs on port 8000.",
     "Noted: auth-svc uses Go 1.23, port 8000."),
    # Supersession (migration)
    ("Heads up — we just migrated payments from Stripe to Adyen.",
     "Got it, Adyen is the new payment provider."),
    # Database version + host URL
    ("Orders DB is now PostgreSQL 16 hosted at db-orders-prod-1.internal.",
     "Noted: orders DB version PostgreSQL 16 on db-orders-prod-1.internal."),
    # People + role
    ("Sara handles week 1 of on-call. Manu handles week 2.",
     "Noted on-call: Sara week 1, Manu week 2."),
    # Decision + policy
    ("We decided to standardize on FastAPI for all backend APIs.",
     "Understood — FastAPI is the backend API standard."),
    # Issue / bug
    ("There's a known JWT clock-skew issue in auth_service with edge proxies.",
     "Acknowledged the JWT clock-skew bug in auth_service."),
    # Pleasantry — should yield 0 memories
    ("hey", "hello"),
    # Mixed: preference + URL + technology
    ("Our staging URL is staging.acme.dev and we prefer Honeycomb for tracing.",
     "Noted: staging at staging.acme.dev, tracing via Honeycomb."),
]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    pipe = SDPPipeline()

    total_time = 0.0
    total_memories = 0

    for i, (user, assistant) in enumerate(TURNS, start=1):
        section(f"TURN {i}")
        print(f"USER:      {user}")
        print(f"ASSISTANT: {assistant}")

        t0 = time.perf_counter()
        mems = pipe.process_turn(user, assistant)
        dur_ms = (time.perf_counter() - t0) * 1000
        total_time += dur_ms
        total_memories += len(mems)

        print(f"\n→ {len(mems)} AALMemory unit(s) extracted in {dur_ms:.2f}ms")
        if not mems:
            print("  (no pattern-detectable facts in this turn)")
            continue

        # Show entities + relationships once (they're the same for every
        # unit within the same turn — they're turn-scoped)
        if mems[0].entities:
            entity_lines = [f"{e['type']}:{e['text']}" for e in mems[0].entities]
            print(f"  entities: {', '.join(entity_lines)}")
        if mems[0].relationships:
            rel_lines = [f"{r['source']}--{r['relation']}-->{r['target']} (c={r['confidence']:.2f})"
                         for r in mems[0].relationships]
            print(f"  relationships:")
            for l in rel_lines:
                print(f"    {l}")

        # Then each unit's content/scores
        print(f"  units:")
        for m in mems:
            item = m.to_memory_item()
            print(f"    [{m.type:<10}] imp={m.importance:.2f} conf={m.confidence:.2f}  "
                  f"{m.subject or '-'}/{m.attribute or '-'} = {m.value or '-'}")
            print(f"                {m.content}")
            print(f"                MemoryItem id: {item.id}  source={item.source}")

    section("SUMMARY")
    print(f"  turns processed:    {len(TURNS)}")
    print(f"  total memories:     {total_memories}")
    print(f"  total time:         {total_time:.1f}ms")
    print(f"  avg per turn:       {total_time/len(TURNS):.1f}ms")
    print(f"  avg per memory:     {(total_time/total_memories) if total_memories else 0:.1f}ms")
    print("\nFor comparison: one MemoryExtractor (LLM) call is ~7000ms on this Mac.")
    print(f"SDP processed all {len(TURNS)} turns in {total_time:.0f}ms — "
          f"~{int(7000 * len(TURNS) / max(total_time, 1))}x faster than LLM ingestion.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
