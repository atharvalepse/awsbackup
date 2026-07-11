"""PgvectorSemanticRetriever — dense retrieval via pgvector cosine distance.

Drop-in for :class:`SemanticRetriever`. Same Retriever ABC, same return
types, very different scale: data lives in Postgres, queries hit the HNSW
index from migration 003 (cosine similarity via the ``<=>`` operator).

Per-user scoping: ``embedded.query.user_id`` (set by the HTTP/MCP handler
after auth) is written into ``app.current_user_id`` inside the transaction
so the RLS policies from migration 005 restrict the query to that user's
memories. When ``user_id`` is None we fall back to admin mode (used by
admin tools + the bench).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from orchestration.errors import RetrieverError
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import (
    EmbeddedQuery, MemoryItem, RetrievalHit,
)
from orchestration.retriever.base import Retriever

if TYPE_CHECKING:
    import asyncpg


slog = StructuredLogger("retriever.pgvector_semantic")


# Match SemanticRetriever's threshold so the YES/NO branch in pipeline.run()
# behaves identically across backends. Lowered 0.30 -> 0.20: keep retrieval
# permissive and let the cross-encoder reranker do quality filtering; 0.30 was
# dropping borderline-relevant matches before the reranker ever saw them.
DEFAULT_MATCH_THRESHOLD = float(os.environ.get("GML_MATCH_THRESHOLD", "0.20"))

# Minimum cosine similarity for a memory to count as a graph NEIGHBOR of a
# seed hit (1-hop expansion). Matches graph_projection's edge threshold so
# "related" means the same thing in retrieval and in the 3D graph view.
DEFAULT_NEIGHBOR_SIM = float(os.environ.get("GML_GRAPH_NEIGHBOR_SIM", "0.45"))


class PgvectorSemanticRetriever(Retriever):
    """Dense retriever backed by pgvector. Uses cosine distance (``<=>``).

    Args:
        pool: asyncpg connection pool (from
            :func:`orchestration.storage.get_pg_pool`).
        match_threshold: minimum similarity (1 - cosine_distance) for
            ``search`` to return a hit. Used for the adversarial gate.

    Design notes
    ------------
    * **No in-memory cache**: every request hits Postgres. HNSW lookup is
      ~1-5 ms even at 1M vectors, so this is fine.
    * **Ingest is a no-op**: the canonical writer is
      :class:`PostgresMemoryStore`, not the retriever. Keeping the
      ``ingest`` method present means HybridRetriever can compose
      pgvector + bm25 without special-casing.
    * **Vector dim mismatch fails fast**: pgvector rejects a 1024-dim
      query against a 384-dim column at the driver level.
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.pool = pool
        self.match_threshold = match_threshold

    # ------------------------------------------------------------------
    # Internal: per-request user scoping via Postgres session vars
    # ------------------------------------------------------------------

    async def _set_session_vars(
        self, conn: "asyncpg.Connection", user_id: str | None
    ) -> None:
        if user_id is None:
            await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
        else:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, true)", user_id
            )

    # ------------------------------------------------------------------
    # Retriever ABC
    # ------------------------------------------------------------------

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        """Cheap probe — top-20 above threshold. Used to pick YES vs NO branch."""
        return await self.get_top_matches(embedded, k=20)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        if not embedded.vector:
            return []

        user_id = embedded.query.user_id
        as_of = getattr(embedded.query, "as_of", None)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._set_session_vars(conn, user_id)
                if as_of is None:
                    # Current beliefs only: open validity interval.
                    rows = await conn.fetch(
                        """
                        SELECT id, content, entity, attribute, value, source,
                               authority_score, pinned, "timestamp",
                               raw_metadata, summary_short, parent_memory_id, entity_id,
                               valid_from, valid_to,
                               1 - (embedding <=> $1) AS similarity
                        FROM memories
                        WHERE embedding IS NOT NULL AND valid_to IS NULL
                        ORDER BY embedding <=> $1
                        LIMIT $2
                        """,
                        embedded.vector,
                        k,
                    )
                else:
                    # Time travel: belief state as of a parameterized instant.
                    rows = await conn.fetch(
                        """
                        SELECT id, content, entity, attribute, value, source,
                               authority_score, pinned, "timestamp",
                               raw_metadata, summary_short, parent_memory_id, entity_id,
                               valid_from, valid_to,
                               1 - (embedding <=> $1) AS similarity
                        FROM memories
                        WHERE embedding IS NOT NULL
                          AND valid_from <= $3
                          AND (valid_to IS NULL OR valid_to > $3)
                        ORDER BY embedding <=> $1
                        LIMIT $2
                        """,
                        embedded.vector,
                        k,
                        as_of,
                    )

        hits: list[RetrievalHit] = []
        for r in rows:
            sim = float(r["similarity"])
            if sim < self.match_threshold:
                # rows come back sorted by distance ascending, so once we
                # drop below threshold everything after is worse too.
                break
            hits.append(RetrievalHit(record=_row_to_item(r), similarity=sim))
        return hits

    async def get_neighbors(
        self, embedded: EmbeddedQuery, record_id: str, k: int = 3
    ) -> list[RetrievalHit]:
        """1-hop graph expansion: the k nearest active memories to the SEED
        record's own embedding (not the query's). These are the same kNN
        edges the visualization draws — consulted here so retrieval can pull
        in facts related to a strong hit that the query embedding alone
        missed. Respects as_of and RLS like get_top_matches."""
        user_id = embedded.query.user_id
        as_of = getattr(embedded.query, "as_of", None)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._set_session_vars(conn, user_id)
                # Force a custom (re-planned-per-call) plan for this
                # transaction. With the parameterized `user_id = $2` qual
                # below, the prepared-statement GENERIC plan can't see tenant
                # selectivity and picks a btree+sort over the whole tenant
                # (idx_memories_current), defeating HNSW → O(N). A custom plan
                # uses the actual value and chooses the HNSW index scan. Scoped
                # to this tx, so it can't affect other queries on the conn.
                await conn.execute(
                    "SELECT set_config('plan_cache_mode','force_custom_plan',true)"
                )
                # Two-step, not a self-join: pulling the seed embedding via a
                # subquery in the kNN SELECT makes the ORDER BY vector a
                # non-constant, which pgvector's HNSW index cannot serve —
                # the planner falls back to a full Nested Loop scan, O(N) per
                # call. Fetch the seed vector first, then pass it as a bound
                # parameter ($1) so the ORDER BY matches the HNSW index.
                seed = await conn.fetchrow(
                    "SELECT embedding, user_id FROM memories WHERE id = $1",
                    record_id,
                )
                if seed is None or seed["embedding"] is None:
                    return []
                seed_vec = seed["embedding"]
                # Explicit tenant scope (defense in depth): RLS also enforces
                # it for app connections, but admin/superuser sessions bypass
                # RLS and must never surface another tenant's memories via a
                # seed id. At scale the planner still serves the ORDER BY from
                # the HNSW index with this equality present (verified by
                # EXPLAIN at 50k); the previous O(N) behaviour came from
                # pulling the seed vector via an in-statement subquery, which
                # made the distance operand non-constant — fixed by the
                # two-step fetch above.
                validity = (
                    "valid_to IS NULL"
                    if as_of is None
                    else "valid_from <= $4 AND (valid_to IS NULL OR valid_to > $4)"
                )
                sql = f"""
                    SELECT id, content, entity, attribute, value,
                           source, authority_score, pinned, "timestamp",
                           raw_metadata, summary_short, parent_memory_id,
                           entity_id, valid_from, valid_to,
                           1 - (embedding <=> $1) AS similarity
                    FROM memories
                    WHERE embedding IS NOT NULL AND user_id = $2
                      AND id <> $3 AND {validity}
                    ORDER BY embedding <=> $1
                    LIMIT {int(k)}
                """
                args = [seed_vec, seed["user_id"], record_id]
                if as_of is not None:
                    args.append(as_of)
                rows = await conn.fetch(sql, *args)

        hits: list[RetrievalHit] = []
        for r in rows:
            sim = float(r["similarity"])
            if sim < DEFAULT_NEIGHBOR_SIM:
                break  # sorted by distance; the rest are weaker
            hits.append(RetrievalHit(record=_row_to_item(r), similarity=sim))
        return hits

    async def ingest(self, records: list[MemoryItem]) -> None:
        """No-op. Memory writes go through :class:`PostgresMemoryStore`.

        Kept on the surface so HybridRetriever can ``await self.dense.ingest(...)``
        without conditional logic. Logs once at debug level if non-empty,
        so misuse is visible in traces.
        """
        if records:
            slog.info(
                event="pgvector_ingest_no_op",
                note="memory persistence is owned by PostgresMemoryStore",
                record_count=len(records),
            )


def _row_to_item(row) -> MemoryItem:
    raw_metadata = row["raw_metadata"]
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except (ValueError, json.JSONDecodeError):
            raw_metadata = {}
    ts = row["timestamp"]
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    keys = set(row.keys())
    valid_from = row["valid_from"] if "valid_from" in keys else None
    valid_to = row["valid_to"] if "valid_to" in keys else None
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        entity=row["entity"],
        attribute=row["attribute"],
        value=row["value"],
        source=row["source"],
        authority_score=float(row["authority_score"]),
        pinned=bool(row["pinned"]),
        timestamp=ts,
        raw_metadata=raw_metadata or {},
        summary_short=row["summary_short"],
        parent_memory_id=row["parent_memory_id"],
        entity_id=row["entity_id"],
        valid_from=valid_from,
        valid_to=valid_to,
        is_latest=valid_to is None,
    )
