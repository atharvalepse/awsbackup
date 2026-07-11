"""Benchmark SAM.resolve_conflicts across multiple Ollama models.

For each model:
  - Build a fresh SAM with that model
  - Run a fixed set of synthetic conflict-resolution cases
  - Measure: latency, correctness (did the expected old-fact get dropped?
    did the new-fact survive?), total drops, total kept
  - Each case has crafted RankedHits with an "old" memory and a "new"
    memory about the same entity+attribute, plus distractors

Print a head-to-head comparison table.

Default models: every locally-pulled non-reasoning model + DeepSeek as
the current production default. Override with --models:

    python scripts/benchmark_sam.py
    python scripts/benchmark_sam.py --models deepseek-r1:8b codestral gemma2:27b
    python scripts/benchmark_sam.py --models qwen2.5:3b llama3.2:3b  # if pulled

Each (model, case) is one Ollama call. With 5 cases × 4 models you get
~20 calls. Expect 30s-3min depending on model sizes.
"""
import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone

from orchestration.pipeline.contracts import (
    MemoryItem,
    Query,
    RankedHit,
    RetrievalHit,
    TargetDescriptor,
)
from orchestration.sam.sam import SAM
from orchestration.sam.llm_reasoner import LLMReasoner
from orchestration.sam._ollama_client import HTTPOllamaClient


def _mk_memory(
    rec_id: str, content: str, entity: str, attribute: str, value: str,
    age_days: float = 0, authority: float = 0.7, source: str = "conversation",
) -> MemoryItem:
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    return MemoryItem(
        id=rec_id,
        content=content,
        entity=entity,
        attribute=attribute,
        value=value,
        timestamp=ts,
        source=source,
        authority_score=authority,
        pinned=False,
    )


def _mk_ranked(mem: MemoryItem, similarity: float, final_score: float) -> RankedHit:
    return RankedHit(
        hit=RetrievalHit(record=mem, similarity=similarity),
        semantic_score=similarity,
        recency_score=1.0,
        authority_score=mem.authority_score,
        pin_boost=0.0,
        final_score=final_score,
        score_reason=f"sim={similarity:.2f} (synthetic)",
    )


# ---------------------------------------------------------------------------
# Synthetic conflict-resolution test cases.
#
# Each case bundles: a user query, a set of ranked memories (mix of old
# and new), the id we expect to be DROPPED, and the id we expect to
# SURVIVE. The "noise" memories are unrelated and should be kept.
# ---------------------------------------------------------------------------


def _make_cases() -> list[dict]:
    cases = []

    # Case A: clear supersession — Stripe → Adyen
    cases.append({
        "id": "A_payments_stripe_to_adyen",
        "query": "what payment provider do we use?",
        "ranked": [
            _mk_ranked(_mk_memory("new-adyen", "Adyen is our payment provider",
                                  entity="payments", attribute="provider", value="Adyen",
                                  age_days=0), 0.95, 0.85),
            _mk_ranked(_mk_memory("old-stripe", "Stripe is our payment provider",
                                  entity="payments", attribute="provider", value="Stripe",
                                  age_days=60), 0.92, 0.55),
            _mk_ranked(_mk_memory("noise-redis", "Redis 7 is our session cache",
                                  entity="session_cache", attribute="version", value="Redis 7",
                                  age_days=20), 0.30, 0.45),
        ],
        "expect_drop": "old-stripe",
        "expect_keep": ["new-adyen", "noise-redis"],
    })

    # Case B: version bump — Postgres 15 → 16
    cases.append({
        "id": "B_postgres_15_to_16",
        "query": "what version of postgres are we on?",
        "ranked": [
            _mk_ranked(_mk_memory("new-pg16", "Orders DB upgraded to PostgreSQL 16",
                                  entity="orders_db", attribute="version", value="16",
                                  age_days=0), 0.93, 0.83),
            _mk_ranked(_mk_memory("old-pg15", "Orders DB is on PostgreSQL 15",
                                  entity="orders_db", attribute="version", value="15",
                                  age_days=90), 0.89, 0.50),
            _mk_ranked(_mk_memory("noise-redis", "Session cache is Redis 7",
                                  entity="session_cache", attribute="version", value="Redis 7",
                                  age_days=20), 0.35, 0.45),
            _mk_ranked(_mk_memory("noise-priya", "Priya leads the payments team",
                                  entity="payments_team", attribute="lead", value="Priya",
                                  age_days=10), 0.20, 0.40),
        ],
        "expect_drop": "old-pg15",
        "expect_keep": ["new-pg16", "noise-redis", "noise-priya"],
    })

    # Case C: NO conflicts — every memory is a distinct fact
    # (SAM should keep ALL; LLM that drops anything is wrong.)
    cases.append({
        "id": "C_no_conflicts_keep_all",
        "query": "tell me about our infra",
        "ranked": [
            _mk_ranked(_mk_memory("infra-1", "auth-svc runs on Go 1.23",
                                  entity="auth_svc", attribute="language", value="Go 1.23",
                                  age_days=2), 0.78, 0.75),
            _mk_ranked(_mk_memory("infra-2", "CI is GitHub Actions",
                                  entity="ci", attribute="tool", value="GitHub Actions",
                                  age_days=15), 0.75, 0.72),
            _mk_ranked(_mk_memory("infra-3", "Deploy uses ArgoCD",
                                  entity="deploy", attribute="tool", value="ArgoCD",
                                  age_days=15), 0.74, 0.71),
            _mk_ranked(_mk_memory("infra-4", "Staging is staging.acme.dev",
                                  entity="staging", attribute="url", value="staging.acme.dev",
                                  age_days=10), 0.72, 0.70),
        ],
        "expect_drop": None,  # nothing should be dropped
        "expect_keep": ["infra-1", "infra-2", "infra-3", "infra-4"],
    })

    # Case D: superseded ON-CALL — Sara → Manu
    cases.append({
        "id": "D_oncall_sara_to_manu",
        "query": "who's on-call this week?",
        "ranked": [
            _mk_ranked(_mk_memory("new-manu", "Manu is full-time on-call",
                                  entity="oncall", attribute="primary", value="Manu",
                                  age_days=1), 0.91, 0.82),
            _mk_ranked(_mk_memory("old-sara1", "Sara handles week 1 of on-call rotation",
                                  entity="oncall", attribute="primary", value="Sara",
                                  age_days=30), 0.87, 0.50),
            _mk_ranked(_mk_memory("noise-deploy", "Deploy uses ArgoCD",
                                  entity="deploy", attribute="tool", value="ArgoCD",
                                  age_days=15), 0.30, 0.42),
        ],
        "expect_drop": "old-sara1",
        "expect_keep": ["new-manu", "noise-deploy"],
    })

    # Case E: trick case — looks like conflict but isn't (different entities)
    cases.append({
        "id": "E_different_entities_not_a_conflict",
        "query": "what languages do we use?",
        "ranked": [
            _mk_ranked(_mk_memory("auth-go", "auth-svc is in Go",
                                  entity="auth_svc", attribute="language", value="Go",
                                  age_days=5), 0.86, 0.78),
            _mk_ranked(_mk_memory("billing-py", "billing-svc is in Python",
                                  entity="billing_svc", attribute="language", value="Python",
                                  age_days=5), 0.85, 0.77),
            _mk_ranked(_mk_memory("api-ts", "api-gateway is in TypeScript",
                                  entity="api_gateway", attribute="language", value="TypeScript",
                                  age_days=5), 0.84, 0.76),
        ],
        "expect_drop": None,  # all different entities — no conflict
        "expect_keep": ["auth-go", "billing-py", "api-ts"],
    })

    return cases


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


async def bench_one(model: str, case: dict) -> dict:
    """Run one SAM call with the given model on one test case. Returns metrics."""
    client = HTTPOllamaClient(model=model, timeout_seconds=60.0)
    reasoner = LLMReasoner(client=client)
    sam = SAM(reasoner=reasoner)
    target = TargetDescriptor.for_claude()
    query = Query(
        text=case["query"], target=target,
        session_context={}, trace_id="bench-" + case["id"],
    )

    t0 = time.perf_counter()
    err = None
    try:
        resolved = await sam.resolve_conflicts(query, case["ranked"])
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        resolved = None
    dur_ms = int((time.perf_counter() - t0) * 1000)

    if resolved is None:
        return {
            "model": model, "case": case["id"], "duration_ms": dur_ms,
            "error": err, "kept_ids": [], "drop_count": 0,
            "expected_drop": case["expect_drop"],
            "correct_drop": False, "kept_expected": False,
            "improved_query": None, "reasoning_chars": 0,
        }

    kept_ids = [
        h.hit.record.id if hasattr(h, "hit") else h.record.id
        for h in resolved.kept
    ]
    ranked_ids = [rh.hit.record.id for rh in case["ranked"]]
    dropped_ids = [rid for rid in ranked_ids if rid not in kept_ids]

    expected_drop = case["expect_drop"]
    correct_drop = (
        (expected_drop is None and len(dropped_ids) == 0)
        or (expected_drop is not None and expected_drop in dropped_ids and len(dropped_ids) == 1)
    )
    kept_expected = all(eid in kept_ids for eid in case["expect_keep"])

    return {
        "model": model,
        "case": case["id"],
        "duration_ms": dur_ms,
        "error": None,
        "kept_ids": kept_ids,
        "drop_count": len(dropped_ids),
        "dropped_ids": dropped_ids,
        "expected_drop": expected_drop,
        "correct_drop": correct_drop,
        "kept_expected": kept_expected,
        "improved_query": resolved.improved_query,
        "reasoning_chars": len(resolved.reasoning_content or ""),
    }


async def run_benchmark(models: list[str]) -> int:
    cases = _make_cases()
    print(f"\n{'=' * 84}")
    print(f"SAM MODEL BENCHMARK — {len(models)} models × {len(cases)} cases = {len(models)*len(cases)} runs")
    print(f"{'=' * 84}")
    for c in cases:
        exp = c["expect_drop"] or "(nothing — keep all)"
        print(f"  · {c['id']:35} query={c['query']!r:55} expect_drop={exp}")
    print(f"{'=' * 84}\n")

    results: list[dict] = []
    for model in models:
        print(f"\n▶ MODEL: {model}")
        for case in cases:
            print(f"  · {case['id']:<38} ", end="", flush=True)
            res = await bench_one(model, case)
            results.append(res)
            if res["error"]:
                print(f"ERROR: {res['error']}")
            else:
                marks = []
                marks.append("✓" if res["correct_drop"] else "✗")
                marks.append("K" if res["kept_expected"] else "k")
                print(f"{res['duration_ms']:>6}ms  drops={res['drop_count']}  {''.join(marks)}  kept={res['kept_ids']}")

    # ---- Summary table ----
    print(f"\n{'=' * 84}")
    print("LEADERBOARD")
    print(f"{'=' * 84}")
    header = f"{'model':<24} {'avg_ms':>8} {'p95_ms':>8} {'correct_drops':>15} {'kept_all_expected':>20}"
    print(header)
    print("─" * 84)
    for model in models:
        rs = [r for r in results if r["model"] == model and not r["error"]]
        if not rs:
            print(f"{model:<24} (all errors)")
            continue
        durations = sorted(r["duration_ms"] for r in rs)
        avg = sum(durations) // len(durations)
        p95 = durations[int(len(durations) * 0.95)] if len(durations) >= 2 else durations[-1]
        correct = sum(1 for r in rs if r["correct_drop"])
        kept = sum(1 for r in rs if r["kept_expected"])
        print(f"{model:<24} {avg:>8} {p95:>8} {correct:>10}/{len(rs):<3} {kept:>15}/{len(rs):<3}")

    # ---- Per-case detail ----
    print(f"\n{'=' * 84}")
    print("PER-CASE DETAIL")
    print(f"{'=' * 84}")
    for case in cases:
        print(f"\n  CASE {case['id']}  query={case['query']!r}")
        print(f"  expected drop: {case['expect_drop']}")
        for model in models:
            r = next((x for x in results if x["model"] == model and x["case"] == case["id"]), None)
            if r is None or r["error"]:
                continue
            verdict = "✓" if r["correct_drop"] and r["kept_expected"] else "✗"
            extra = ""
            if not r["correct_drop"]:
                extra = f" (actually dropped: {r['dropped_ids']})"
            print(f"    {verdict} {model:<24} {r['duration_ms']:>6}ms  drops={r['drop_count']}{extra}")
            if r["improved_query"] and r["improved_query"] != case["query"]:
                snippet = r["improved_query"][:90]
                print(f"        improved_query: {snippet!r}")

    print()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+",
        default=["deepseek-r1:8b", "codestral:latest", "gemma2:27b"],
        help="Ollama model names to benchmark (default: deepseek + codestral + gemma2:27b)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(run_benchmark(args.models)))
