"""Hybrid retriever — fuses dense + BM25 results via Reciprocal Rank Fusion.

RRF is the standard lexical+semantic combiner used in production search
stacks (Vespa, Elasticsearch, Vertex). It's:

  score(d) = Σ_r  1 / (k + rank_r(d))

over each ranker ``r``. ``k`` damps high-rank dominance; LOCOMO-shaped
workloads benefit most from ``k`` ≈ 60 (the original Cormack et al. tuning).

Practically: dense retrieval captures semantic intent, BM25 catches
exact-term recall (rare entity names, version strings, identifiers).
Fusing the two consistently lifts top-k recall by 10-20% on memory
benchmarks vs either alone.
"""
import asyncio

from orchestration.pipeline.contracts import EmbeddedQuery, MemoryItem, RetrievalHit
from orchestration.retriever.base import Retriever


DEFAULT_RRF_K = 60


class HybridRetriever(Retriever):
    """RRF-fused dense + BM25 retriever.

    Both inner retrievers run in parallel. The fused score per record is
    a normalized RRF value in [0, 1] — usable as similarity downstream.

    Records ingested via :meth:`ingest` flow to both inner retrievers (BM25
    via its sync ``ingest``, dense via its async ``ingest`` if available).

    Args:
        dense: a Retriever with an async ``ingest`` (e.g. SemanticRetriever).
        sparse: a BM25Retriever or any Retriever with a sync ``ingest``.
        rrf_k: damping constant; 60 is the canonical default.
        candidates_per_retriever: how many hits each inner retriever returns
            before fusion. 100 is a good ceiling for most corpora.
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        rrf_k: int = DEFAULT_RRF_K,
        candidates_per_retriever: int = 100,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.candidates_per_retriever = candidates_per_retriever

    async def ingest(self, records: list[MemoryItem]) -> None:
        """Pass records to both inner retrievers. Either ``ingest`` may be sync
        (in-memory BM25 / SemanticRetriever) or async (the pgvector-backed
        variants), so await whichever returns a coroutine — otherwise the async
        one is never awaited (the 'coroutine was never awaited' warning)."""
        for inner in (self.sparse, self.dense):
            if hasattr(inner, "ingest"):
                res = inner.ingest(records)
                if asyncio.iscoroutine(res):
                    await res

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        return await self._fuse(embedded, k=self.candidates_per_retriever)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        return await self._fuse(embedded, k=k)

    async def _fuse(self, embedded: EmbeddedQuery, k: int) -> list[RetrievalHit]:
        dense_task = asyncio.create_task(
            self.dense.get_top_matches(embedded, k=self.candidates_per_retriever)
        )
        sparse_task = asyncio.create_task(
            self.sparse.get_top_matches(embedded, k=self.candidates_per_retriever)
        )
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

        # RRF: each ranker contributes 1/(k + rank). Score is sum.
        # Keep one MemoryItem instance per id (prefer dense's instance for
        # the canonical record body).
        records_by_id: dict[str, MemoryItem] = {}
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            records_by_id[hit.record.id] = hit.record
            rrf_scores[hit.record.id] = (
                rrf_scores.get(hit.record.id, 0.0) + 1.0 / (self.rrf_k + rank)
            )

        for rank, hit in enumerate(sparse_hits, start=1):
            records_by_id.setdefault(hit.record.id, hit.record)
            rrf_scores[hit.record.id] = (
                rrf_scores.get(hit.record.id, 0.0) + 1.0 / (self.rrf_k + rank)
            )

        if not rrf_scores:
            return []

        # Normalize RRF scores to [0, 1] so downstream stages treat the
        # fused similarity uniformly with raw cosines.
        max_score = max(rrf_scores.values())
        fused = [
            RetrievalHit(
                record=records_by_id[id_],
                similarity=score / max_score if max_score > 0 else 0.0,
            )
            for id_, score in rrf_scores.items()
        ]
        fused.sort(key=lambda h: h.similarity, reverse=True)
        return fused[:k]
