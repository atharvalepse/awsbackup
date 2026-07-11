#!/usr/bin/env python3
"""Speed audit: context-storing (ingest) and retrieval latency for the
current orchestration code, against a local pgvector cluster.

Isolates the DB + orchestration path (what recent perf work changed) from
fixed embedder/reranker model cost by using precomputed clustered vectors.
Clustered (not uniform-random) so cosine similarity has a realistic
distribution and queries return real above-threshold hits.

DSN via GML_TEST_DATABASE_URL.
"""
import asyncio
import json
import os
import statistics
import time
import uuid

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from orchestration.pipeline.contracts import EmbeddedQuery, MemoryItem, Query, TargetDescriptor
from orchestration.pipeline.contracts import Classification, ClassificationSource
from orchestration.retriever.pgvector_semantic import PgvectorSemanticRetriever
from orchestration.storage.postgres_memory_store import PostgresMemoryStore

DSN = os.environ["GML_TEST_DATABASE_URL"]
DIM = 384
N_CLUSTERS = 256
rng = np.random.default_rng(7)
CENTERS = rng.standard_normal((N_CLUSTERS, DIM))
CENTERS /= np.linalg.norm(CENTERS, axis=1, keepdims=True)


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def mem_vec(i):
    c = CENTERS[i % N_CLUSTERS]
    return _unit(c + 0.35 * rng.standard_normal(DIM))


def query_vec():
    c = CENTERS[rng.integers(N_CLUSTERS)]
    return _unit(c + 0.15 * rng.standard_normal(DIM)).tolist()


def pct(xs, p):
    return round(statistics.quantiles(xs, n=100)[p - 1], 2)


def report(name, xs):
    return {
        "op": name, "n": len(xs),
        "p50_ms": pct(xs, 50), "p95_ms": pct(xs, 95),
        "p99_ms": pct(xs, 99), "max_ms": round(max(xs), 2),
    }


async def seed(pool, user_id, start, count):
    """Bulk-insert clustered rows directly (bypassing the gate) to build a
    corpus fast. Returns wall seconds."""
    rows = []
    now = "2026-01-01T00:00:00+00:00"
    for i in range(start, start + count):
        v = mem_vec(i).tolist()
        rows.append((
            f"seed-{user_id[-6:]}-{i}", user_id, f"memory content number {i}",
            len(f"memory content number {i}"), v,
        ))
    t0 = time.perf_counter()
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.is_admin','true',false)")
        await conn.execute(
            "INSERT INTO users (user_id, plan, quota_bytes) "
            "VALUES ($1,'free',10737418240) ON CONFLICT (user_id) DO NOTHING",
            user_id)
        await conn.executemany(
            """INSERT INTO memories (id, user_id, content, byte_size, embedding,
                   valid_from)
               VALUES ($1,$2,$3,$4,$5, TIMESTAMPTZ '2026-01-01')
               ON CONFLICT (id) DO NOTHING""",
            rows,
        )
    elapsed = time.perf_counter() - t0
    # Settle background work (HNSW build, autovacuum, checkpoint) so it
    # doesn't pollute the latency measurement that follows.
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.is_admin','true',false)")
        await conn.execute("CHECKPOINT")
        await conn.execute("ANALYZE memories")
    await asyncio.sleep(1.0)
    return elapsed


async def bench_retrieval(retriever, user_id, n_queries=200):
    cls = Classification(intent_type="question", confidence=0.5,
                         source=ClassificationSource.KEYWORD_FALLBACK)
    target = TargetDescriptor.for_chatgpt()
    probe, full = [], []
    for _ in range(n_queries):
        q = Query(text="bench", target=target, user_id=user_id, trace_id="t")
        eq = EmbeddedQuery(query=q, classification=cls, vector=query_vec(),
                           embedder_version="bench")
        t = time.perf_counter()
        await retriever.search(eq)              # probe k=20
        probe.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        await retriever.get_top_matches(eq, k=50)
        full.append((time.perf_counter() - t) * 1000)
    return report("retrieval_probe_k20", probe), report("retrieval_top50", full)


async def bench_neighbors(retriever, pool, user_id, n=100):
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.is_admin','true',false)")
        ids = [r["id"] for r in await conn.fetch(
            "SELECT id FROM memories WHERE user_id=$1 LIMIT $2", user_id, n)]
    cls = Classification(intent_type="question", confidence=0.5,
                         source=ClassificationSource.KEYWORD_FALLBACK)
    q = Query(text="bench", target=TargetDescriptor.for_chatgpt(), user_id=user_id, trace_id="t")
    eq = EmbeddedQuery(query=q, classification=cls, vector=query_vec(),
                       embedder_version="bench")
    lat = []
    for mid in ids:
        t = time.perf_counter()
        await retriever.get_neighbors(eq, mid, k=3)
        lat.append((time.perf_counter() - t) * 1000)
    return report("graph_neighbors_k3", lat)


async def bench_as_of(retriever, user_id, n=100):
    from datetime import datetime, timezone
    cls = Classification(intent_type="question", confidence=0.5,
                         source=ClassificationSource.KEYWORD_FALLBACK)
    as_of = datetime(2026, 3, 1, tzinfo=timezone.utc)
    lat = []
    for _ in range(n):
        q = Query(text="bench", target=TargetDescriptor.for_chatgpt(),
                  user_id=user_id, as_of=as_of, trace_id="t")
        eq = EmbeddedQuery(query=q, classification=cls, vector=query_vec(),
                           embedder_version="bench")
        t = time.perf_counter()
        await retriever.get_top_matches(eq, k=50)
        lat.append((time.perf_counter() - t) * 1000)
    return report("retrieval_as_of_top50", lat)


def make_items(prefix, n, *, entity_mode):
    """entity_mode: 'novel' = each a fresh entity (full resolver+gate);
    'known' = all reuse one entity (pre-pass hit after first)."""
    items = []
    for i in range(n):
        if entity_mode == "novel":
            ent = f"Service {prefix} {i}"
        else:
            ent = "Shared Platform Service"
        items.append(MemoryItem(
            id=f"ing-{prefix}-{uuid.uuid4().hex[:10]}",
            content=f"{prefix} fact {i} about the platform",
            entity=ent, attribute="status", value=f"v{i}",
            timestamp=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc),
            source="bench", authority_score=0.6,
            raw_metadata={"_embedding": mem_vec(i + 900000).tolist(),
                          "session_id": f"sess-{prefix}"},
        ))
    return items


async def bench_ingest(store, label, items, batch=50):
    """Time add_many in chunks; return throughput + per-batch latency."""
    uid = "ingbench-" + uuid.uuid4().hex[:8]
    lat = []
    t0 = time.perf_counter()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        t = time.perf_counter()
        await store.add_many(chunk, user_id=uid)
        lat.append((time.perf_counter() - t) * 1000)
    total = time.perf_counter() - t0
    return {
        "scenario": label, "items": len(items), "batch_size": batch,
        "total_s": round(total, 3),
        "throughput_per_s": round(len(items) / total, 1),
        "per_batch_p50_ms": pct(lat, 50) if len(lat) > 1 else round(lat[0], 2),
        "per_item_ms": round(total / len(items) * 1000, 3),
    }, uid


async def main():
    pool = await asyncpg.create_pool(DSN, min_size=2, max_size=4,
                                     init=register_vector)
    out = {"corpus_retrieval": [], "ingest": []}
    try:
        retr = PgvectorSemanticRetriever(pool)
        store = PostgresMemoryStore(pool)
        # Fixed user + deterministic row ids so re-runs reuse the seeded
        # corpus (ON CONFLICT DO NOTHING makes re-seed a fast no-op).
        bench_user = "retrbench-fixed"

        # ---- Retrieval at growing corpus sizes -------------------------
        sizes = [1000, 10000, 50000]
        seeded = 0
        for target_n in sizes:
            add = target_n - seeded
            seed_s = await seed(pool, bench_user, seeded, add)
            seeded = target_n
            # warm HNSW
            for _ in range(20):
                cls = Classification(intent_type="question", confidence=0.5,
                                     source=ClassificationSource.KEYWORD_FALLBACK)
                q = Query(text="w", target=TargetDescriptor.for_chatgpt(),
                          user_id=bench_user, trace_id="t")
                await retr.get_top_matches(
                    EmbeddedQuery(query=q, classification=cls,
                                  vector=query_vec(), embedder_version="b"), k=50)
            probe_r, full_r = await bench_retrieval(retr, bench_user)
            nbr_r = await bench_neighbors(retr, pool, bench_user)
            asof_r = await bench_as_of(retr, bench_user)
            print(f"[corpus={target_n}] seed {add} rows in {seed_s:.1f}s "
                  f"({add/seed_s:.0f}/s) | top50 p50={full_r['p50_ms']}ms "
                  f"p99={full_r['p99_ms']}ms")
            out["corpus_retrieval"].append({
                "corpus_size": target_n,
                "seed_rows": add, "seed_seconds": round(seed_s, 2),
                "seed_throughput_per_s": round(add / seed_s, 0),
                "results": [probe_r, full_r, nbr_r, asof_r],
            })

        # ---- Ingest (context storing) through the real write path ------
        os.environ["GML_GATE_LOG"] = "on"
        r1, _ = await bench_ingest(store, "novel_entities_gate_on_log_on",
                                   make_items("nov", 300, entity_mode="novel"))
        r2, _ = await bench_ingest(store, "known_entity_prepass_gate_on",
                                   make_items("kno", 300, entity_mode="known"))
        os.environ["GML_GATE_LOG"] = "off"
        r3, _ = await bench_ingest(store, "novel_entities_gate_log_off",
                                   make_items("nl2", 300, entity_mode="novel"))
        os.environ["GML_WRITE_GATE"] = "off"
        r4, _ = await bench_ingest(store, "novel_entities_gate_off",
                                   make_items("ng", 300, entity_mode="novel"))
        os.environ.pop("GML_WRITE_GATE", None)
        os.environ["GML_GATE_LOG"] = "on"
        out["ingest"] = [r1, r2, r3, r4]
        for r in out["ingest"]:
            print(f"[ingest] {r['scenario']}: {r['throughput_per_s']}/s "
                  f"({r['per_item_ms']}ms/item)")

        # cleanup ingest bench rows only; keep the retrieval corpus so
        # repeated runs reuse it (pass --clean to wipe everything).
        async with pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.is_admin','true',false)")
            await conn.execute("DELETE FROM memories WHERE user_id LIKE 'ingbench-%'")
            if "--clean" in __import__("sys").argv:
                await conn.execute(
                    "DELETE FROM memories WHERE user_id LIKE 'retrbench-%'")
    finally:
        await pool.close()

    with open("audit_speed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote audit_speed.json")


if __name__ == "__main__":
    asyncio.run(main())
