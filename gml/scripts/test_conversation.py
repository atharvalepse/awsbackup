"""Adversarial multi-turn conversation test for the GML pipeline.

Simulates what Claude Desktop does: each user turn calls `query()` to get
context, then `ingest()` to persist new facts. The conversation is designed
to STRESS the pipeline:

  1. State facts (Turns 1-3)            — exercises ingest + extractor
  2. Supersede those facts (Turns 4-5)  — exercises SAM conflict resolution
  3. Ask back questions (Turns 6+)      — exercises retrieval, reranker,
                                          SAM (forced to choose new vs old)
  4. Vague reference (Turn 10)          — exercises SAM query improvement
  5. Cross-domain question (Turn 11)    — exercises retrieval across topics

After each turn, prints stage-by-stage timing + key outputs. At the end,
prints summary stats (total turns, total time, SAM skip rate, …).

Run:
    cd /Users/atharvalepse/Projects/gml-orchestration
    .venv/bin/python scripts/test_conversation.py

Or with environment overrides:
    GML_FORCE_SAM=1 .venv/bin/python scripts/test_conversation.py
    GML_SAM_SKIP_THRESHOLD=0.5 .venv/bin/python scripts/test_conversation.py
"""
import asyncio
import os
import re
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Conversation script — designed to exercise every stage of the pipeline.
# Each entry is either ("ingest", user_text, assistant_reply) or
# ("query", user_text, expected_keywords_in_context).
# ---------------------------------------------------------------------------

CONVERSATION = [
    # --- Phase 1: state initial facts (no conflicts yet) -------------
    ("ingest",
     "Just so you know, our payment service runs on Stripe and the team lead is Priya Iyer.",
     "Got it — noted Stripe as the payments provider and Priya Iyer as the payments team lead."),

    ("ingest",
     "We use PostgreSQL 15 for the orders database. It runs on db-orders-prod-1.internal.",
     "Noted: orders DB is PostgreSQL 15, host db-orders-prod-1.internal."),

    ("ingest",
     "The api-gateway service is deployed on Kubernetes 1.27 in cluster prod-eu-west.",
     "Noted: api-gateway runs on Kubernetes 1.27 in cluster prod-eu-west."),

    # --- Phase 2: state contradicting / superseding facts ------------
    # These should force SAM into a real conflict-resolution call.
    ("ingest",
     "Update: we finished migrating payments from Stripe to Adyen yesterday — Adyen is the new payment provider.",
     "Understood — payments provider has changed from Stripe to Adyen."),

    ("ingest",
     "Just upgraded the orders database to PostgreSQL 16 last weekend.",
     "Noted: orders DB version is now PostgreSQL 16."),

    # --- Phase 3: questions that force retrieval + (maybe) SAM -------
    ("query",
     "What payment provider do we use?",
     ["Adyen"]),  # MUST surface the newer fact

    ("query",
     "What version of PostgreSQL is the orders database on?",
     ["16"]),  # MUST surface the newer version

    ("query",
     "Who runs the payments team?",
     ["Priya"]),  # Unambiguous — should skip SAM

    ("query",
     "What kubernetes version does api-gateway use?",
     ["1.27"]),  # Unambiguous — should skip SAM

    # --- Phase 4: stress tests ---------------------------------------
    ("query",
     "What was that issue we had with auth_service?",
     ["JWT", "clock-skew"]),  # vague reference

    ("query",
     "What database engine do we use?",
     ["PostgreSQL"]),  # ambiguous — might pull orders + others

    ("query",
     "Where does the payments provider currently route?",
     ["Adyen"]),  # follow-up to Q1
]


def _trace_to_summary(trace_text: str) -> dict:
    """Extract per-stage timing from a `trace()` output. Returns a dict
    of stage_name -> duration_ms, plus key indicators."""
    summary = {"stages": [], "sam_skipped": False, "sam_kept": None, "assembled": None}
    for line in trace_text.splitlines():
        m = re.match(r"\[\d+\]\s+([A-Z_.]+).*?\((\d+)ms\)", line)
        if m:
            summary["stages"].append({"name": m.group(1), "ms": int(m.group(2))})
        if "kept" in line and ":" in line:
            mk = re.search(r"kept\s*:\s*(\d+)", line)
            if mk and summary["sam_kept"] is None:
                summary["sam_kept"] = int(mk.group(1))
        if "selected" in line and ":" in line:
            ms = re.search(r"selected\s*:\s*(\d+)", line)
            if ms and summary["assembled"] is None:
                summary["assembled"] = int(ms.group(1))
    return summary


def _first_text(call_tool_result) -> str:
    for c in call_tool_result.content:
        if hasattr(c, "text"):
            return c.text
    return ""


def _check_keywords(context: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Return (all_found, missing_keywords)."""
    text = context.lower()
    missing = [k for k in keywords if k.lower() not in text]
    return len(missing) == 0, missing


HORIZ = "─" * 78


async def run_test() -> int:
    print(f"\n{'=' * 78}")
    print("GML — ADVERSARIAL CONVERSATION TEST")
    print(f"{'=' * 78}")
    print(f"  GML_FORCE_SAM={os.environ.get('GML_FORCE_SAM', '(unset → SAM skip enabled)')}")
    print(f"  GML_SAM_SKIP_THRESHOLD={os.environ.get('GML_SAM_SKIP_THRESHOLD', '(default 0.75)')}")
    print(f"  Turns: {len(CONVERSATION)}")
    print(f"{'=' * 78}\n")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "orchestration.mcp_server"],
    )

    stats = {
        "total_turns": 0,
        "total_query_ms": 0,
        "total_ingest_ms": 0,
        "sam_skipped_count": 0,
        "sam_ran_count": 0,
        "passes": 0,
        "fails": 0,
        "failed_turns": [],
    }

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()

            # Show pre-test state
            print("--- INITIAL STATE ---")
            diag_res = await session.call_tool("diag", {})
            print(_first_text(diag_res))
            print()

            for i, turn in enumerate(CONVERSATION, start=1):
                kind = turn[0]
                stats["total_turns"] += 1
                print(HORIZ)
                if kind == "ingest":
                    _, user_text, assistant_reply = turn
                    print(f"[TURN {i}] INGEST")
                    print(f"  user:  {user_text}")
                    print(f"  reply: {assistant_reply}")
                    t0 = time.perf_counter()
                    res = await session.call_tool("ingest", {
                        "user_query": user_text,
                        "assistant_reply": assistant_reply,
                    })
                    dur = int((time.perf_counter() - t0) * 1000)
                    stats["total_ingest_ms"] += dur
                    print(f"  → ingest ({dur}ms): {_first_text(res)}")

                elif kind == "query":
                    _, user_text, expected = turn
                    print(f"[TURN {i}] QUERY")
                    print(f"  user: {user_text}")
                    print(f"  expecting context to contain: {expected}")

                    # Use trace() so we see every stage
                    t0 = time.perf_counter()
                    trace_res = await session.call_tool("trace", {"text": user_text})
                    dur = int((time.perf_counter() - t0) * 1000)
                    stats["total_query_ms"] += dur
                    trace_text = _first_text(trace_res)

                    # Extract just the stages section + formatted_context
                    summary = _trace_to_summary(trace_text)
                    sam_stage = next(
                        (s for s in summary["stages"] if s["name"].startswith("SAM") or "SAM" in s["name"]),
                        None,
                    )
                    sam_skipped = "SKIPPED" in trace_text or "SAM skipped" in trace_text
                    if sam_skipped:
                        stats["sam_skipped_count"] += 1
                    else:
                        stats["sam_ran_count"] += 1

                    # Stage timing one-liner
                    stage_line = "  stages: " + " → ".join(
                        f"{s['name'].lower().replace('.', '_')[:15]}:{s['ms']}ms"
                        for s in summary["stages"]
                    )
                    print(stage_line)
                    print(f"  total: {dur}ms  sam_kept={summary['sam_kept']}  assembled={summary['assembled']}  sam_skipped={sam_skipped}")

                    # Check expected keywords are in the formatted context
                    fc_match = re.search(r"--- formatted_context.*?---\n(.*)$", trace_text, re.DOTALL)
                    fc = fc_match.group(1) if fc_match else ""
                    passed, missing = _check_keywords(fc, expected)
                    if passed:
                        stats["passes"] += 1
                        print(f"  ✓ PASS — all expected keywords found in context")
                    else:
                        stats["fails"] += 1
                        stats["failed_turns"].append((i, user_text, missing))
                        print(f"  ✗ FAIL — missing keywords: {missing}")
                        print(f"  context preview (first 300 chars):")
                        print("    " + fc[:300].replace("\n", "\n    "))

                print()

            # Final state
            print(HORIZ)
            print("--- FINAL STATE ---")
            diag_res = await session.call_tool("diag", {})
            print(_first_text(diag_res))

    # Summary
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Turns:                  {stats['total_turns']}")
    print(f"  Passes:                 {stats['passes']}")
    print(f"  Fails:                  {stats['fails']}")
    print(f"  Total query time:       {stats['total_query_ms']/1000:.1f}s")
    print(f"  Total ingest time:      {stats['total_ingest_ms']/1000:.1f}s")
    print(f"  SAM ran (LLM):          {stats['sam_ran_count']} time(s)")
    print(f"  SAM skipped:            {stats['sam_skipped_count']} time(s)")
    sam_total = stats['sam_ran_count'] + stats['sam_skipped_count']
    if sam_total:
        print(f"  SAM skip rate:          {stats['sam_skipped_count']*100//sam_total}%")
    if stats["failed_turns"]:
        print(f"\n  Failed turns:")
        for i, q, missing in stats["failed_turns"]:
            print(f"    Turn {i}: {q!r}  missing {missing}")
    print(f"{'=' * 78}\n")

    return 0 if stats["fails"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_test()))
