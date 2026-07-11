"""TimeAwareRetriever — temporal-question bias on top of any base retriever.

Phase B7 + B8 from the LOCOMO improvement plan.

Behavior
--------
When the query is classified as TEMPORAL (``QueryHints.is_temporal``):
  1. Pull candidates from the base retriever as normal.
  2. Apply a similarity ``boost`` to any candidate whose CONTENT contains
     at least one date pattern (uses ``date_extractor.has_date``). This is
     Phase B8 ("answer-grounded scoring") — when the question asks "when",
     candidates containing dates are more likely to hold the answer.
  3. Re-sort by the boosted similarity.

For non-temporal queries, behave as a pure pass-through wrapper.

Wraps any ``Retriever`` (HybridRetriever, EntityBoostedRetriever, etc.)
and mirrors the interface so it can be composed anywhere.
"""
from orchestration.pipeline.contracts import EmbeddedQuery, MemoryItem, RetrievalHit
from orchestration.retriever.base import Retriever
from orchestration.sdp.date_extractor import has_date
from orchestration.sdp.query_router import classify_query


DEFAULT_DATE_BOOST = 0.15


class TimeAwareRetriever(Retriever):
    """Boost similarity of date-bearing candidates on temporal queries."""

    def __init__(
        self,
        base: Retriever,
        date_boost: float = DEFAULT_DATE_BOOST,
    ) -> None:
        self.base = base
        self.date_boost = date_boost

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        hits = await self.base.search(embedded)
        return self._maybe_boost(hits, embedded.query.text)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        hits = await self.base.get_top_matches(embedded, k=k)
        return self._maybe_boost(hits, embedded.query.text)[:k]

    async def ingest(self, items: list[MemoryItem]) -> None:
        if hasattr(self.base, "ingest"):
            await self.base.ingest(items)

    # ----------------------------------------------------------------------

    def _maybe_boost(
        self, hits: list[RetrievalHit], query_text: str
    ) -> list[RetrievalHit]:
        if not hits:
            return hits
        hints = classify_query(query_text)
        if not hints.is_temporal:
            return hits

        boosted: list[RetrievalHit] = []
        for h in hits:
            if has_date(h.record.content):
                new_sim = min(1.0, h.similarity + self.date_boost)
                boosted.append(RetrievalHit(record=h.record, similarity=new_sim))
            else:
                boosted.append(h)
        boosted.sort(key=lambda x: -x.similarity)
        return boosted
