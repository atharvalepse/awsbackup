"""Ad-hoc verification of embedding-on-write (PostgresMemoryStore + embedder).

Distinct from postgres_smoke_test.py: that test PRE-embeds the row. This one
adds memories with NO preset _embedding and checks the store fills the vector
column itself. Run against a throwaway DB:

    GML_DATABASE_URL='postgresql://gml_app:PASS@127.0.0.1:5432/gml_test' \
        .venv/bin/python scripts/verify_embed_on_write.py
"""
import asyncio
import sys
import time
import uuid
from datetime import datetime, timezone

from orchestration.embedder import FastEmbedEmbedder
from orchestration.pipeline.contracts import (
    Classification, ClassificationSource, EmbeddedQuery, Query, TargetDescriptor,
)
from orchestration.retriever.pgvector_semantic import PgvectorSemanticRetriever
from orchestration.storage import close_pg_pool, get_pg_pool
from orchestration.storage.postgres_memory_store import PostgresMemoryStore

G, R, X = "\033[32m", "\033[31m", "\033[0m"
PASS = f"{G}✓{X}"
FAIL = f"{R}✗{X}"
failures = 0


def check(cond, msg):
    global failures
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        failures += 1


def memory_item(content, source="verify"):
    from orchestration.pipeline.contracts import MemoryItem
    return MemoryItem(
        id=f"eow-{uuid.uuid4().hex[:12]}",
        content=content,
        entity=None, attribute=None, value=None,
        source=source, authority_score=0.7, pinned=False,
        timestamp=datetime.now(timezone.utc),
        raw_metadata={"note": "keep me"},  # non-embedding metadata must survive
        summary_short=None,
    )


async def main():
    pool = await get_pg_pool()
    embedder = FastEmbedEmbedder()
    uid = f"eow_{int(time.time())}"

    try:
        # tenant
        async with pool.acquire() as c:
            await c.execute(
                "INSERT INTO users (user_id, plan, quota_bytes) VALUES ($1,'free',1073741824) "
                "ON CONFLICT (user_id) DO UPDATE SET bytes_used=0",
                uid,
            )

        # ---- 1) WITH embedder, NO preset _embedding ----------------------
        print("\n[1] add via embedder-backed store (no preset vector)")
        store = PostgresMemoryStore(pool, embedder=embedder)
        m_pay = memory_item("We migrated billing to Stripe for payments.")
        m_cloud = memory_item("Production runs on Google Cloud in us-central1.")
        await store.add_many([m_pay, m_cloud], user_id=uid)

        async with pool.acquire() as c:
            async with c.transaction():  # set_config(...,true) is txn-local
                await c.execute("SELECT set_config('app.current_user_id', $1, true)", uid)
                row = await c.fetchrow(
                    "SELECT embedding IS NOT NULL AS has_vec, raw_metadata "
                    "FROM memories WHERE id=$1", m_pay.id,
                )
        check(row["has_vec"], "embedding column populated by the store (was NULL before fix)")
        import json as _json
        meta = row["raw_metadata"]
        meta = _json.loads(meta) if isinstance(meta, str) else meta
        check("_embedding" not in (meta or {}), "vector NOT duplicated inside raw_metadata JSONB")
        check((meta or {}).get("note") == "keep me", "other raw_metadata survives")

        # ---- 2) retrievable via pgvector cosine --------------------------
        print("\n[2] semantic retrieval finds the embedded-on-write row")
        retr = PgvectorSemanticRetriever(pool)
        qvec = (await embedder.embed_batch(["what payment provider do we use?"]))[0]
        q = Query(text="what payment provider do we use?",
                  target=TargetDescriptor.for_claude(),
                  trace_id=uuid.uuid4().hex, user_id=uid)
        eq = EmbeddedQuery(
            query=q,
            classification=Classification(
                intent_type="question", entities=[], retrieval_hints={},
                confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK),
            vector=qvec, embedder_version="verify")
        hits = await retr.get_top_matches(eq, k=5)
        check(len(hits) >= 1, f"got {len(hits)} hit(s)")
        check(bool(hits) and hits[0].record.id == m_pay.id,
              f"top hit is the payments memory (sim={hits[0].similarity:.3f})" if hits else "no hits")

        # ---- 3) NO embedder, NO preset -> NULL embedding -----------------
        print("\n[3] store WITHOUT embedder leaves embedding NULL (+ warns)")
        store_noemb = PostgresMemoryStore(pool)  # no embedder
        m_null = memory_item("This row should land with a NULL embedding.")
        await store_noemb.add_many([m_null], user_id=uid)
        async with pool.acquire() as c:
            async with c.transaction():
                await c.execute("SELECT set_config('app.current_user_id', $1, true)", uid)
                has_vec = await c.fetchval(
                    "SELECT embedding IS NOT NULL FROM memories WHERE id=$1", m_null.id)
        check(has_vec is False, "embedding is NULL when no embedder is wired in")

    finally:
        async with pool.acquire() as c:
            await c.execute("SELECT set_config('app.is_admin','true',true)")
            await c.execute("DELETE FROM users WHERE user_id=$1", uid)
        await close_pg_pool()

    print()
    if failures:
        print(f"{R}{failures} check(s) FAILED.{X}")
        return 1
    print(f"{G}embedding-on-write verified — all checks passed.{X}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
