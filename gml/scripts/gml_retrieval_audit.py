"""GML retrieval audit harness — black-box probes against a running API.

Probes (each emits pass/fail + evidence into a JSON scorecard):
  P1 entity duplication     — alias variants stored as distinct entities; dup claims crowd top-k
  P2 extraction coverage    — dense multi-fact turn → how many facts did SDP capture?
  P3 noise vs signal        — low-value chatter outranking high-value facts
  P4 staleness              — superseded fact still surfaces as current
  P5 retrieval legs         — exact entity/attr + rare-token lookups (vector-only blind spots)
  P6 ranking composition    — is ranking sensitive to anything beyond similarity?
  P7 write determinism      — same conflicting write-set, shuffled, N tenants → same final state?
  P8 conflict honesty       — two contradicting active claims: are both surfaced/flagged?
  LAT latency               — p50/p95 for recall + synthesize

Usage:
  python scripts/gml_retrieval_audit.py --base http://localhost:8000 \
      --out audit_scorecard.json

Each run signs up fresh `audit+<nonce>@gml.test` tenants so probes are
RLS-isolated from real data. Probe tenants are left behind (cheap, inert);
pass --cleanup to delete their memories afterwards.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import string
import sys
import time

import httpx

DEFAULT_BASE = "http://localhost:8000"
PASSWORD = "audit-probe-pw-1"

# Deployed servers may gate /auth/signup behind invite codes. Load a pool of
# unused codes (one per probe tenant) via --invite-codes <file>, one per line.
INVITE_CODES: list[str] = []


def nonce(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


class Tenant:
    """One signed-up probe user with its own bearer token."""

    def __init__(self, client: httpx.AsyncClient, token: str, email: str):
        self.client = client
        self.token = token
        self.email = email

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    @classmethod
    async def create(cls, client: httpx.AsyncClient) -> "Tenant":
        email = f"audit+{nonce()}@gml.test"
        body = {"email": email, "password": PASSWORD}
        if INVITE_CODES:
            body["invite_code"] = INVITE_CODES.pop()
        r = await client.post("/auth/signup", json=body)
        r.raise_for_status()
        return cls(client, r.json()["access_token"], email)

    async def sdp_ingest(self, user_query: str, assistant_reply: str) -> dict:
        r = await self.client.post(
            "/api/memory/sdp_ingest",
            json={"user_query": user_query, "assistant_reply": assistant_reply},
            headers=self.headers,
        )
        r.raise_for_status()
        return r.json()

    async def recall(
        self, query: str, top_k: int = 10, as_of: str | None = None
    ) -> list[dict]:
        body = {"query": query, "top_k": top_k}
        if as_of is not None:
            body["as_of"] = as_of
        r = await self.client.post(
            "/api/memory/recall", json=body, headers=self.headers,
        )
        r.raise_for_status()
        return r.json()["results"]

    async def synthesize(self, query: str) -> dict:
        r = await self.client.get(
            "/api/memory/synthesize", params={"query": query}, headers=self.headers
        )
        r.raise_for_status()
        return r.json()

    async def memories(self) -> list[dict]:
        out, offset = [], 0
        while True:
            r = await self.client.get(
                "/api/memories",
                params={"limit": 500, "offset": offset},
                headers=self.headers,
            )
            r.raise_for_status()
            page = r.json()
            out.extend(page["memories"])
            offset += len(page["memories"])
            if offset >= page["total"] or not page["memories"]:
                return out

    async def cleanup(self) -> None:
        for m in await self.memories():
            await self.client.delete(f"/api/memories/{m['id']}", headers=self.headers)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


async def p1_entity_duplication(client) -> dict:
    t = await Tenant.create(client)
    turns = [
        ("Tell me about the project",
         "GML is a memory layer for multi-agent systems built on FastAPI."),
        ("What stack does it use?",
         "The Gigzs Multi-LLM Layer uses PostgreSQL with pgvector for storage."),
        ("Anything else about the system?",
         "GML stores memories as claims. GML runs on FastAPI and uses PostgreSQL."),
    ]
    for uq, ar in turns:
        await t.sdp_ingest(uq, ar)
    mems = await t.memories()
    entities = sorted({(m.get("entity") or "").strip().lower() for m in mems} - {""})
    gml_aliases = [e for e in entities if "gml" in e or "gigzs" in e or "multi-llm" in e]
    hits = await t.recall("what is GML built on", top_k=10)
    contents = [h["memory"]["content"] for h in hits]
    # near-duplicate top-k: same (entity, attribute, value) appearing more than once
    keys = [
        (h["memory"].get("entity"), h["memory"].get("attribute"), h["memory"].get("value"))
        for h in hits
    ]
    dup_in_topk = len(keys) - len(set(keys))
    return {
        "probe": "P1_entity_duplication",
        "alias_entities_for_same_subject": gml_aliases,
        "n_alias_entities": len(gml_aliases),
        "duplicate_claims_in_top10": dup_in_topk,
        "top10_contents": contents,
        "passed": len(gml_aliases) <= 1 and dup_in_topk == 0,
    }


DENSE_TURN = (
    "Summarize our architecture decisions",
    "We decided the arbitration layer must stay neutral across OpenAI, Anthropic "
    "and Google. Priya owns the belief-revision router. The importance floor for "
    "ingest is 0.4, conflicts escalate to the truth dashboard instead of being "
    "guessed, and superseded claims keep a forward link rather than being deleted. "
    "We're targeting a p95 recall latency of 150ms on Postgres 15.",
)
DENSE_TURN_EXPECTED = [
    "arbitration neutrality across providers",
    "priya owns belief-revision router",
    "importance floor 0.4",
    "conflicts escalate to truth dashboard",
    "superseded claims keep forward link",
    "p95 latency target 150ms",
]


async def p2_extraction_coverage(client) -> dict:
    t = await Tenant.create(client)
    res = await t.sdp_ingest(*DENSE_TURN)
    mems = await t.memories()
    return {
        "probe": "P2_extraction_coverage",
        "expected_facts": len(DENSE_TURN_EXPECTED),
        "claims_stored": len(mems),
        "stored_contents": [m["content"] for m in mems],
        "ingest_detail": res.get("detail"),
        "passed": len(mems) >= 4,  # at least 4 of the 6 dense facts captured
    }


async def p3_noise_vs_signal(client) -> dict:
    t = await Tenant.create(client)
    # signal
    await t.sdp_ingest(
        "Record this decision",
        "The production database password rotation runs via Kubernetes on port 5432 "
        "and the deploy authority is Shivansh.",
    )
    # noise
    for i in range(5):
        await t.sdp_ingest(
            "hey", f"sounds good, thanks! have a great day, talk soon {i}"
        )
        await t.sdp_ingest("ok", f"no worries at all, happy to help anytime {i}")
    mems = await t.memories()
    noise_stored = [
        m["content"] for m in mems
        if "thanks" in m["content"].lower() or "no worries" in m["content"].lower()
    ]
    hits = await t.recall("database deploy details", top_k=5)
    top5 = [h["memory"]["content"] for h in hits]
    noise_in_top5 = sum(
        1 for c in top5 if "thanks" in c.lower() or "no worries" in c.lower()
    )
    return {
        "probe": "P3_noise_vs_signal",
        "noise_claims_stored": len(noise_stored),
        "noise_in_top5": noise_in_top5,
        "top5": top5,
        "passed": len(noise_stored) == 0 and noise_in_top5 == 0,
    }


async def p4_staleness(client) -> dict:
    t = await Tenant.create(client)
    await t.sdp_ingest(
        "Note my setup", "Our staging deploy runs on Heroku with Redis 6."
    )
    await asyncio.sleep(0.2)
    await t.sdp_ingest(
        "Update: we migrated", "We moved off Heroku; staging now runs on GCP with Redis 7."
    )
    hits = await t.recall("where does staging run", top_k=5)
    top = [h["memory"]["content"] for h in hits]
    stale_rank = next((i for i, c in enumerate(top) if "Heroku" in c and "GCP" not in c), None)
    fresh_rank = next((i for i, c in enumerate(top) if "GCP" in c), None)
    mems = await t.memories()
    return {
        "probe": "P4_staleness",
        "total_claims_stored": len(mems),
        "stale_rank": stale_rank,
        "fresh_rank": fresh_rank,
        "top5": top,
        # pass = fresh fact exists and outranks stale, and stale is not presented
        # as co-equal current truth at rank 0
        "passed": fresh_rank is not None
        and (stale_rank is None or fresh_rank < stale_rank),
    }


async def p5_retrieval_legs(client) -> dict:
    t = await Tenant.create(client)
    await t.sdp_ingest(
        "Save this", "The license key for Vault is XK99-QJZ7. Marek manages the Vault cluster.",
    )
    await t.sdp_ingest(
        "And this", "Our internal codename for the reranker project is Operation Walnut.",
    )
    results = {}
    for label, q, needle in [
        ("rare_token", "XK99-QJZ7", "XK99"),
        ("exact_entity", "who manages Vault", "Marek"),
        ("codename", "Operation Walnut", "Walnut"),
    ]:
        hits = await t.recall(q, top_k=5)
        found = any(needle.lower() in h["memory"]["content"].lower() for h in hits)
        results[label] = {"query": q, "found": found, "n_hits": len(hits)}
    return {
        "probe": "P5_retrieval_legs",
        "lookups": results,
        "passed": all(r["found"] for r in results.values()),
    }


async def p6_ranking_composition(client) -> dict:
    """Two near-identical claims, one old / one new: does anything besides raw
    similarity influence order? (Recency should break the tie.)"""
    t = await Tenant.create(client)
    await t.sdp_ingest("note", "The build server is Jenkins running on port 8080.")
    await asyncio.sleep(0.3)
    await t.sdp_ingest("note again", "The build server is Jenkins on port 8080 still.")
    hits = await t.recall("build server", top_k=5)
    scores = [h["score"] for h in hits]
    distinct_scores = len(set(scores)) == len(scores)
    return {
        "probe": "P6_ranking_composition",
        "scores": scores,
        "note": "raw /recall path returns retriever similarity only; composite "
                "rerank lives in the /synthesize pipeline",
        "passed": distinct_scores and len(scores) >= 2,
    }


CONFLICT_SET = [
    ("status update", "The API gateway timeout is set to 30 seconds."),
    ("status update", "The API gateway timeout is set to 60 seconds."),
    ("status update", "The API gateway timeout is set to 90 seconds."),
    ("owner update", "Riya owns the gateway config."),
    ("owner update", "Dev owns the gateway config."),
]


async def _final_state_for_shuffled_writes(client, order: list[int]) -> list[str]:
    t = await Tenant.create(client)
    # concurrent conflicting writes in the given order-batches
    await asyncio.gather(*(t.sdp_ingest(*CONFLICT_SET[i]) for i in order))
    mems = await t.memories()
    return sorted(m["content"] for m in mems)


async def p7_write_determinism(client, trials: int = 4) -> dict:
    rng = random.Random(42)
    states = []
    for _ in range(trials):
        order = list(range(len(CONFLICT_SET)))
        rng.shuffle(order)
        states.append(await _final_state_for_shuffled_writes(client, order))
    canonical = states[0]
    identical = all(s == canonical for s in states)
    return {
        "probe": "P7_write_determinism",
        "trials": trials,
        "identical_final_states": identical,
        "state_sizes": [len(s) for s in states],
        "example_state": canonical,
        "passed": identical,
    }


async def p8_conflict_honesty(client) -> dict:
    t = await Tenant.create(client)
    await t.sdp_ingest("config check", "The rate limit is 100 requests per minute.")
    await t.sdp_ingest("config check", "The rate limit is 500 requests per minute.")
    hits = await t.recall("what is the rate limit", top_k=5)
    contents = [h["memory"]["content"] for h in hits]
    both_surfaced = any("100" in c for c in contents) and any("500" in c for c in contents)
    flagged = any(
        "conflict" in json.dumps(h).lower() for h in hits
    )
    synth = await t.synthesize("what is the rate limit")
    ctx = synth.get("context", "")
    synth_both = ("100" in ctx) and ("500" in ctx)
    synth_flagged = "conflict" in ctx.lower() or "contradic" in ctx.lower()
    return {
        "probe": "P8_conflict_honesty",
        "recall_both_surfaced": both_surfaced,
        "recall_conflict_flag": flagged,
        "synthesize_both_present": synth_both,
        "synthesize_flagged": synth_flagged,
        "synth_context": ctx[:500],
        "passed": both_surfaced and (flagged or synth_flagged),
    }


async def p9_staleness_hard(client) -> dict:
    """Superseded fact must be ABSENT from default recall — not just ranked low."""
    t = await Tenant.create(client)
    await t.sdp_ingest("Note my setup", "Our staging deploy runs on Heroku with Redis 6.")
    await asyncio.sleep(0.3)
    await t.sdp_ingest(
        "Update: we migrated",
        "We moved off Heroku; staging now runs on GCP with Redis 7.",
    )
    hits = await t.recall("where does staging run", top_k=10)
    contents = [h["memory"]["content"] for h in hits]
    def _is_stale(c: str) -> bool:
        # a positive Heroku assertion is stale; a retraction ("no longer",
        # "moved off") is a correct current belief
        low = c.lower()
        return ("heroku" in low and "gcp" not in low
                and "no longer" not in low and "moved off" not in low)

    stale = [c for c in contents if _is_stale(c)]
    fresh = [c for c in contents if "GCP" in c]
    return {
        "probe": "P9_staleness_hard",
        "stale_claims_in_recall": stale,
        "fresh_claims_in_recall": fresh,
        "all": contents,
        "passed": bool(fresh) and not stale,
    }


async def p10_time_travel(client) -> dict:
    """As-of query returns the old belief; default query returns the new one."""
    from datetime import datetime, timezone
    t = await Tenant.create(client)
    await t.sdp_ingest("Note my setup", "Our staging deploy runs on Heroku with Redis 6.")
    await asyncio.sleep(0.5)
    t_between = datetime.now(timezone.utc).isoformat()
    await asyncio.sleep(0.5)
    await t.sdp_ingest(
        "Update: we migrated",
        "We moved off Heroku; staging now runs on GCP with Redis 7.",
    )
    past = [h["memory"]["content"]
            for h in await t.recall("where does staging run", top_k=10, as_of=t_between)]
    cur = [h["memory"]["content"]
           for h in await t.recall("where does staging run", top_k=10)]
    def _pos_heroku(c: str) -> bool:
        low = c.lower()
        return ("heroku" in low and "gcp" not in low
                and "no longer" not in low and "moved off" not in low)

    past_has_old = any(_pos_heroku(c) for c in past)
    past_has_new = any("GCP" in c for c in past)
    cur_has_new = any("GCP" in c for c in cur)
    cur_has_old = any(_pos_heroku(c) for c in cur)
    return {
        "probe": "P10_time_travel",
        "as_of": t_between,
        "past_results": past,
        "current_results": cur,
        "passed": past_has_old and not past_has_new and cur_has_new and not cur_has_old,
    }


async def p11_entity_resolution(client) -> dict:
    """Alias mentions ("Priya" / "Priya Sharma") must share one canonical
    entity_id at write time — the fix for duplicate-entity crowding."""
    t = await Tenant.create(client)
    await t.sdp_ingest(
        "Who owns the router?", "Priya owns the belief-revision router."
    )
    await t.sdp_ingest(
        "Who manages deploys?", "Priya Sharma manages the deployment pipeline."
    )
    mems = await t.memories()
    by_alias = {
        m["entity"]: m.get("entity_id")
        for m in mems
        if (m.get("entity") or "").lower().startswith("priya")
    }
    ids = {v for v in by_alias.values() if v}
    return {
        "probe": "P11_entity_resolution",
        "alias_to_entity_id": by_alias,
        "distinct_entity_ids": len(ids),
        "passed": len(by_alias) >= 2 and len(ids) == 1
        and all(v for v in by_alias.values()),
    }


async def latency(client) -> dict:
    t = await Tenant.create(client)
    await t.sdp_ingest("warm", "Warmup claim: PostgreSQL runs on port 5432.")
    recall_ms, synth_ms, ingest_ms = [], [], []
    for i in range(10):
        t0 = time.perf_counter()
        await t.sdp_ingest(
            f"note {i}", f"The batch-{i} worker runs on port {7000 + i}."
        )
        ingest_ms.append((time.perf_counter() - t0) * 1000)
    for _ in range(10):
        t0 = time.perf_counter()
        await t.recall("postgres port", top_k=5)
        recall_ms.append((time.perf_counter() - t0) * 1000)
    for _ in range(5):
        t0 = time.perf_counter()
        await t.synthesize("postgres port")
        synth_ms.append((time.perf_counter() - t0) * 1000)

    def stats(xs):
        xs = sorted(xs)
        return {
            "p50_ms": round(statistics.median(xs), 1),
            "p95_ms": round(xs[max(0, int(len(xs) * 0.95) - 1)], 1),
            "max_ms": round(xs[-1], 1),
        }

    return {
        "probe": "LATENCY",
        "ingest": stats(ingest_ms),
        "recall": stats(recall_ms),
        "synthesize": stats(synth_ms),
        # ingest guards the write path (Phase 2 baseline ~58ms p50); recall
        # bound is loose because the reranked path is CPU-heavy on this box.
        "passed": statistics.median(ingest_ms) < 150
        and statistics.median(recall_ms) < 1500,
    }


PROBES = [
    p1_entity_duplication,
    p2_extraction_coverage,
    p3_noise_vs_signal,
    p4_staleness,
    p5_retrieval_legs,
    p6_ranking_composition,
    p7_write_determinism,
    p8_conflict_honesty,
    p9_staleness_hard,
    p10_time_travel,
    p11_entity_resolution,
    latency,
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default="audit_scorecard.json")
    ap.add_argument("--invite-codes", help="file of unused invite codes, one per line")
    args = ap.parse_args()
    if args.invite_codes:
        with open(args.invite_codes) as f:
            INVITE_CODES.extend(line.strip() for line in f if line.strip())

    async with httpx.AsyncClient(base_url=args.base, timeout=60.0) as client:
        r = await client.get("/api/health")
        r.raise_for_status()
        health = r.json()
        results = []
        for probe in PROBES:
            name = probe.__name__
            try:
                res = await probe(client)
            except Exception as exc:
                res = {"probe": name, "error": f"{type(exc).__name__}: {exc}", "passed": False}
            results.append(res)
            status = "PASS" if res.get("passed") else "FAIL"
            extra = f" ({res['error']})" if "error" in res else ""
            print(f"[{status}] {res.get('probe', name)}{extra}")

    scorecard = {
        "base": args.base,
        "server": health,
        "results": results,
        "n_pass": sum(1 for r in results if r.get("passed")),
        "n_total": len(results),
    }
    with open(args.out, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nscorecard → {args.out}  ({scorecard['n_pass']}/{scorecard['n_total']} passed)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
