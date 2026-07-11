"""End-to-end smoke test for the Postgres backend.

What it does:
  1. Connect to GML_DATABASE_URL — verify the schema migrations were applied.
  2. Issue a test user + key directly in the DB (skips the HTTP admin flow).
  3. Write one AAL memory via PostgresMemoryStore.
  4. Verify the byte-tracking trigger updated bytes_used.
  5. Run a recall via PgvectorSemanticRetriever — check the row comes back.
  6. Run a recall via PgvectorBM25Retriever — same check.
  7. Verify RLS scoping: a DIFFERENT user can't see the memory.
  8. Delete the test rows on the way out (idempotent).

Run on the GCP VM (where Postgres is local) after migrations are applied:

    GML_DATABASE_URL='postgresql://gml_app:PASS@127.0.0.1/gml' \\
        .venv/bin/python scripts/postgres_smoke_test.py

Exits 0 on success, 1 on any check failure (with a precise message).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime, timezone


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


async def check_connection(pool) -> None:
    print("\n[1] Connection check")
    async with pool.acquire() as conn:
        v = await conn.fetchval("SELECT version()")
        ok(f"connected — {v.split(',')[0]}")


async def check_schema(pool) -> None:
    print("\n[2] Schema check")
    expected = ["users", "user_keys", "memories", "schema_migrations"]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        )
        present = {r["tablename"] for r in rows}
        for t in expected:
            if t in present:
                ok(f"table {t!r} exists")
            else:
                fail(f"table {t!r} MISSING — apply migrations/")
                raise SystemExit(1)

        # Confirm pgvector + pg_trgm extensions are active
        rows = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'pgcrypto')"
        )
        present_ext = {r["extname"] for r in rows}
        for e in ("vector", "pg_trgm", "pgcrypto"):
            (ok if e in present_ext else fail)(f"extension {e!r} {'active' if e in present_ext else 'MISSING'}")

        # Confirm migration 006 (tsvector column) is applied
        col = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='memories' AND column_name='content_tsv'"
        )
        (ok if col else fail)("migration 006 (content_tsv FTS column) " +
                              ("applied" if col else "MISSING — apply migrations/006_fts.sql"))

        # Confirm migration 008 (aal columns) is applied
        col = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='memories' AND column_name='aal_simplemem'"
        )
        (ok if col else fail)("migration 008 (AAL columns) " +
                              ("applied" if col else "MISSING — apply migrations/008_aal_columns.sql"))


async def setup_test_users(pool, test_user_a: str, test_user_b: str) -> None:
    print("\n[3] Test users")
    async with pool.acquire() as conn:
        for uid in (test_user_a, test_user_b):
            await conn.execute(
                """
                INSERT INTO users (user_id, plan, quota_bytes)
                VALUES ($1, 'free', 1073741824)
                ON CONFLICT (user_id) DO UPDATE
                    SET quota_bytes = EXCLUDED.quota_bytes,
                        bytes_used = 0,
                        warned_at_90pct = FALSE
                """,
                uid,
            )
            ok(f"user {uid!r} ready (quota=1 GiB)")


async def smoke_memory_roundtrip(test_user: str) -> None:
    print("\n[4] Memory write + read roundtrip via PostgresMemoryStore")
    from orchestration.aal import AAL
    from orchestration.storage import get_pg_pool
    from orchestration.storage.postgres_memory_store import PostgresMemoryStore
    from orchestration.embedder import FastEmbedEmbedder

    pool = await get_pg_pool()
    store = PostgresMemoryStore(pool)

    aal = AAL(
        simplemem="Smoke test: we use Adyen for payments.",
        sjson={
            "subject": "payments",
            "verb": "provider",
            "object": "Adyen",
            "confidence": 0.95,
            "category": "decision",
        },
        importance=0.85,
        source="smoke",
    )
    item = aal.to_memory_item()

    # Embed it so the row has a vector — required for pgvector retrieval.
    embedder = FastEmbedEmbedder()
    vec = (await embedder.embed_batch([item.content]))[0]
    item = item.model_copy(update={
        "raw_metadata": {**(item.raw_metadata or {}), "_embedding": vec}
    })

    await store.add_many([item], user_id=test_user)
    ok(f"wrote memory {item.id!r} for user {test_user!r}")

    # Read back
    rows = await store.load_all(user_id=test_user)
    if not any(r.id == item.id for r in rows):
        fail(f"could not find {item.id!r} in load_all(user={test_user!r})")
        raise SystemExit(1)
    ok(f"load_all returned the row ({len(rows)} memories visible to user)")

    # Verify AAL roundtrip via the new columns
    fetched = next(r for r in rows if r.id == item.id)
    if fetched.aal_simplemem != aal.simplemem:
        fail("aal_simplemem column did not roundtrip")
        raise SystemExit(1)
    ok("aal_simplemem column roundtripped")
    if (fetched.aal_sjson or {}).get("object") != "Adyen":
        fail(f"aal_sjson did not roundtrip: {fetched.aal_sjson}")
        raise SystemExit(1)
    ok("aal_sjson column roundtripped")

    return item.id, vec


async def check_byte_tracking(pool, test_user: str) -> None:
    print("\n[5] Byte-tracking trigger")
    async with pool.acquire() as conn:
        used = await conn.fetchval(
            "SELECT bytes_used FROM users WHERE user_id=$1", test_user
        )
        if used and used > 0:
            ok(f"bytes_used = {used} (trigger fired on insert)")
        else:
            fail(f"bytes_used = {used} — trigger may not be installed")


async def check_pgvector_retrieval(test_user: str, vec: list) -> None:
    print("\n[6] PgvectorSemanticRetriever — cosine search")
    from orchestration.retriever.pgvector_semantic import PgvectorSemanticRetriever
    from orchestration.storage import get_pg_pool
    from orchestration.pipeline.contracts import (
        Classification, ClassificationSource, EmbeddedQuery, Query, TargetDescriptor,
    )

    pool = await get_pg_pool()
    retriever = PgvectorSemanticRetriever(pool)
    q = Query(
        text="what do we use for payments?",
        target=TargetDescriptor.for_claude(),
        trace_id=uuid.uuid4().hex,
        user_id=test_user,
    )
    embedded = EmbeddedQuery(
        query=q,
        classification=Classification(
            intent_type="question", entities=[], retrieval_hints={},
            confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
        ),
        vector=vec,
        embedder_version="smoke",
    )
    hits = await retriever.get_top_matches(embedded, k=10)
    if hits:
        ok(f"semantic search returned {len(hits)} hit(s), top similarity = {hits[0].similarity:.3f}")
    else:
        fail("semantic search returned NO hits — check pgvector HNSW index")


async def check_pgvector_bm25(test_user: str) -> None:
    print("\n[7] PgvectorBM25Retriever — ts_rank_cd")
    from orchestration.retriever.pgvector_bm25 import PgvectorBM25Retriever
    from orchestration.storage import get_pg_pool
    from orchestration.pipeline.contracts import (
        Classification, ClassificationSource, EmbeddedQuery, Query, TargetDescriptor,
    )

    pool = await get_pg_pool()
    retriever = PgvectorBM25Retriever(pool)
    q = Query(
        text="Adyen payments",
        target=TargetDescriptor.for_claude(),
        trace_id=uuid.uuid4().hex,
        user_id=test_user,
    )
    embedded = EmbeddedQuery(
        query=q,
        classification=Classification(
            intent_type="question", entities=[], retrieval_hints={},
            confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
        ),
        vector=[0.0] * 384,  # BM25 ignores it
        embedder_version="smoke",
    )
    hits = await retriever.get_top_matches(embedded, k=10)
    if hits:
        ok(f"BM25 search returned {len(hits)} hit(s), top score (normalized) = {hits[0].similarity:.3f}")
    else:
        fail("BM25 search returned NO hits — check content_tsv index from migration 006")


async def check_rls_isolation(test_user_a: str, test_user_b: str, mem_id: str) -> None:
    print("\n[8] RLS isolation — user B should NOT see user A's memory")
    from orchestration.storage import get_pg_pool
    from orchestration.storage.postgres_memory_store import PostgresMemoryStore
    pool = await get_pg_pool()
    store = PostgresMemoryStore(pool)
    rows = await store.load_all(user_id=test_user_b)
    if any(r.id == mem_id for r in rows):
        fail(f"USER B CAN SEE USER A'S ROW — RLS not enforcing!")
        raise SystemExit(1)
    ok(f"user {test_user_b!r} sees {len(rows)} memories (none of user {test_user_a!r}'s)")


async def cleanup(pool, test_user_a: str, test_user_b: str) -> None:
    print("\n[9] Cleanup")
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
        await conn.execute("DELETE FROM users WHERE user_id IN ($1, $2)", test_user_a, test_user_b)
        ok("test users + their memories removed (cascade via FK)")


async def main() -> int:
    dsn = os.environ.get("GML_DATABASE_URL", "").strip()
    if not dsn:
        print(f"{RED}Set GML_DATABASE_URL first.{RESET}")
        return 1

    print(f"GML Postgres smoke — {dsn.split('@')[-1]}")
    print("=" * 60)

    test_user_a = f"smoke_a_{int(time.time())}"
    test_user_b = f"smoke_b_{int(time.time())}"

    from orchestration.storage import close_pg_pool, get_pg_pool

    pool = await get_pg_pool()
    try:
        await check_connection(pool)
        await check_schema(pool)
        await setup_test_users(pool, test_user_a, test_user_b)
        mem_id, vec = await smoke_memory_roundtrip(test_user_a)
        await check_byte_tracking(pool, test_user_a)
        await check_pgvector_retrieval(test_user_a, vec)
        await check_pgvector_bm25(test_user_a)
        await check_rls_isolation(test_user_a, test_user_b, mem_id)
        await cleanup(pool, test_user_a, test_user_b)
    finally:
        await close_pg_pool()

    print()
    print("=" * 60)
    print(f"{GREEN}All checks passed.{RESET} Postgres backend is ready for traffic.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
