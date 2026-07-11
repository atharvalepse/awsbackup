"""PgvectorBM25Retriever — sparse retrieval via Postgres tsvector.

Drop-in for :class:`BM25Retriever`. The in-memory BM25Okapi we used before
required full corpus rebuild on every restart and lived only in one
process. The Postgres ``content_tsv`` column (migration 006) + GIN index
give us BM25-like ranking with ``ts_rank_cd`` natively — scales to
millions of rows, persists across restarts, multi-process safe.

Per-user scoping: same pattern as the pgvector semantic retriever — read
``embedded.query.user_id`` and write it to the Postgres session var so
RLS gates the result.

Note on semantics: ``ts_rank_cd`` returns a *relevance score* that's NOT
in [0, 1]. We normalize by min-max within each query's result set so the
HybridRetriever's RRF fusion behaves the same as it did with BM25Okapi.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import (
    EmbeddedQuery, MemoryItem, RetrievalHit,
)
from orchestration.retriever.base import Retriever
from orchestration.retriever.pgvector_semantic import _row_to_item

if TYPE_CHECKING:
    import asyncpg


slog = StructuredLogger("retriever.pgvector_bm25")


# Strip characters that would confuse plainto_tsquery. Keep alphanumerics
# + whitespace + a few common punctuation marks that tsvector handles.
_QUERY_CLEAN_RE = re.compile(r"[^\w\s\-]")


def _clean_query(text: str) -> str:
    """Sanitize the query for plainto_tsquery — strip exotic punctuation
    that the parser would reject."""
    return _QUERY_CLEAN_RE.sub(" ", text or "").strip()


class PgvectorBM25Retriever(Retriever):
    """Sparse retriever using Postgres tsvector + ``ts_rank_cd``.

    Composes with :class:`HybridRetriever` exactly the same way
    :class:`BM25Retriever` did — same Retriever ABC, same RetrievalHit
    output shape. HybridRetriever's RRF fusion remains unchanged.
    """

    def __init__(self, pool: "asyncpg.Pool") -> None:
        self.pool = pool

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
        # Lexical retrieval has no notion of an "adversarial threshold" —
        # tsvector either matches the query or doesn't. Just return top-50.
        return await self.get_top_matches(embedded, k=50)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        q_text = _clean_query(embedded.query.text)
        if not q_text:
            return []

        user_id = embedded.query.user_id
        as_of = getattr(embedded.query, "as_of", None)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._set_session_vars(conn, user_id)
                if as_of is None:
                    rows = await conn.fetch(
                        """
                        SELECT id, content, entity, attribute, value, source,
                               authority_score, pinned, "timestamp",
                               raw_metadata, summary_short, parent_memory_id, entity_id,
                               ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
                        FROM memories
                        WHERE content_tsv @@ plainto_tsquery('english', $1)
                          AND valid_to IS NULL
                        ORDER BY rank DESC
                        LIMIT $2
                        """,
                        q_text,
                        k,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, content, entity, attribute, value, source,
                               authority_score, pinned, "timestamp",
                               raw_metadata, summary_short, parent_memory_id, entity_id,
                               ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
                        FROM memories
                        WHERE content_tsv @@ plainto_tsquery('english', $1)
                          AND valid_from <= $3
                          AND (valid_to IS NULL OR valid_to > $3)
                        ORDER BY rank DESC
                        LIMIT $2
                        """,
                        q_text,
                        k,
                        as_of,
                    )

        if not rows:
            return []

        # ts_rank_cd output is unbounded above. Min-max normalize within
        # the result set so HybridRetriever's RRF treats it on equal
        # footing with the dense [0,1] scores. The normalized score is
        # NOT a probability — it's just a within-page ordering signal,
        # which is what RRF needs.
        raw_ranks = [float(r["rank"]) for r in rows]
        rmax = max(raw_ranks) or 1.0
        rmin = min(raw_ranks)
        span = (rmax - rmin) or 1.0

        return [
            RetrievalHit(
                record=_row_to_item(r),
                similarity=(float(r["rank"]) - rmin) / span,
            )
            for r in rows
        ]

    async def ingest(self, records: list[MemoryItem]) -> None:
        """No-op. ``content_tsv`` is a generated column maintained by
        Postgres — writes go through :class:`PostgresMemoryStore`."""
        if records:
            slog.info(
                event="pg_bm25_ingest_no_op",
                note="content_tsv is auto-maintained by Postgres",
                record_count=len(records),
            )
