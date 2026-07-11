"""Storage backend factory + connection management.

Two backends live behind the same interfaces:

  * **JSONL** (existing): ``~/.gml/memories.jsonl`` + ``~/.gml/users.jsonl``.
    Zero infra. Single-tenant. Dev / fallback only.
  * **Postgres** (new): row-level-security per user, pgvector for dense
    retrieval, ``tsvector`` for sparse, byte-tracking trigger for quota.
    Production target.

Backend is picked at startup by ``GML_STORAGE_BACKEND`` env var
(``"jsonl"`` default, ``"postgres"`` opts in).

Connection management
---------------------
Postgres connections are pooled by ``asyncpg``. The pool is a per-process
singleton, created lazily on first use. ``register_vector`` is run on
every newly-acquired connection so ``vector(384)`` columns work natively.
"""
from __future__ import annotations

import os
from pathlib import Path

from orchestration.memory_store.base import MemoryStore
from orchestration.memory_store.jsonl_store import JsonlMemoryStore
from orchestration.users import UserKeyStore as JsonlUserKeyStore


def _backend() -> str:
    return os.environ.get("GML_STORAGE_BACKEND", "jsonl").strip().lower()


def _is_postgres() -> bool:
    return _backend() == "postgres"


# Lazy singletons. Avoid importing asyncpg at module load — it's only
# needed when the Postgres backend is selected.
_pg_pool = None


async def get_pg_pool():
    """Return the process-wide asyncpg pool. Creates it on first call.

    Raises ``RuntimeError`` if the Postgres backend is selected but
    ``GML_DATABASE_URL`` is unset (config error — fail loud at startup
    rather than per-request later).
    """
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    dsn = os.environ.get("GML_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "GML_STORAGE_BACKEND=postgres but GML_DATABASE_URL is unset. "
            "Set it to a Postgres DSN like "
            "postgresql://gml_app:PASS@127.0.0.1/gml"
        )

    import asyncpg
    from pgvector.asyncpg import register_vector

    async def _init(conn):
        await register_vector(conn)

    _pg_pool = await asyncpg.create_pool(
        dsn,
        min_size=int(os.environ.get("GML_DB_POOL_MIN", "1")),
        max_size=int(os.environ.get("GML_DB_POOL_MAX", "10")),
        init=_init,
        command_timeout=30,
    )
    return _pg_pool


async def close_pg_pool() -> None:
    """Close the asyncpg pool on app shutdown."""
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


async def make_memory_store(
    fallback_jsonl_path: str | Path | None = None,
    embedder=None,
) -> MemoryStore:
    """Build a MemoryStore for the current backend.

    For Postgres: returns ``PostgresMemoryStore`` bound to the pool. Pass the
    process embedder so it can populate the pgvector ``embedding`` column on
    write — without it, ingested rows land with NULL embeddings and are never
    vector-retrievable.
    For JSONL: returns ``JsonlMemoryStore`` at ``fallback_jsonl_path``
    (or ``~/.gml/memories.jsonl`` when unset). The embedder is ignored — the
    in-memory retriever embeds at ingest time.
    """
    if _is_postgres():
        from orchestration.storage.postgres_memory_store import PostgresMemoryStore
        pool = await get_pg_pool()
        return PostgresMemoryStore(pool, embedder=embedder)

    path = fallback_jsonl_path or Path.home() / ".gml" / "memories.jsonl"
    return JsonlMemoryStore(path)


async def make_user_key_store():
    """Build a UserKeyStore for the current backend.

    Both stores expose the same async-friendly API
    (``lookup``, ``issue``, ``revoke``, ``list_users``).
    """
    if _is_postgres():
        from orchestration.storage.postgres_user_store import PostgresUserKeyStore
        pool = await get_pg_pool()
        return PostgresUserKeyStore(pool)
    return JsonlUserKeyStore()


async def make_hybrid_retriever(embedder):
    """Build the dense+sparse HybridRetriever for the current backend.

    Postgres: PgvectorSemanticRetriever (cosine via pgvector) +
              PgvectorBM25Retriever (ts_rank_cd).
    JSONL:    SemanticRetriever (in-memory vectors) +
              BM25Retriever (in-memory BM25Okapi).

    Both return a :class:`HybridRetriever`-conformant object so every
    downstream stage (EntityBoosted, MultiHopAware, TimeAware, the
    reranker chain) composes identically.
    """
    from orchestration.retriever.hybrid_retriever import HybridRetriever
    if _is_postgres():
        from orchestration.retriever.pgvector_bm25 import PgvectorBM25Retriever
        from orchestration.retriever.pgvector_semantic import PgvectorSemanticRetriever
        pool = await get_pg_pool()
        # Optional cosine floor for dense retrieval. This is the correct home
        # for the recall "junk floor" (true cosine), unlike a filter on the
        # fused RRF score, which decays by rank and is not a cosine value.
        # Opt-in: when GML_RECALL_SIM_FLOOR is unset the default (0.30) is kept
        # so prod stays aligned with the JSONL/bench SemanticRetriever and the
        # pipeline's cross-backend YES/NO gate. Set e.g. 0.55 in prod to drop
        # superficial 0.4-0.5 matches.
        _floor = os.environ.get("GML_RECALL_SIM_FLOOR")
        dense = (
            PgvectorSemanticRetriever(pool, match_threshold=float(_floor))
            if _floor is not None
            else PgvectorSemanticRetriever(pool)
        )
        return HybridRetriever(
            dense=dense,
            sparse=PgvectorBM25Retriever(pool),
        )
    from orchestration.retriever.bm25_retriever import BM25Retriever
    from orchestration.retriever.semantic_retriever import SemanticRetriever
    return HybridRetriever(
        dense=SemanticRetriever(embedder=embedder),
        sparse=BM25Retriever(),
    )


__all__ = [
    "make_memory_store",
    "make_user_key_store",
    "make_hybrid_retriever",
    "get_pg_pool",
    "close_pg_pool",
]
