"""EntityBoostedRetriever — wraps any Retriever with an entity prior.

When the user asks *"What did Caroline research?"* and Caroline is an
entity we've indexed, we want memories that mention Caroline to win ties
against memories that don't — even if cosine similarity slightly favors
the unrelated ones.

Mechanism: maintain an :class:`EntityIndex` alongside the base retriever.
On every retrieval, extract entities from the query text. For each hit
whose record_id is in the indexed entity's memory set, bump its
similarity by ``boost`` (default 0.15). Re-sort. Return.

If no entities are found in the query, fall through to the base retriever
unchanged. Failure of entity extraction never hurts retrieval — only
helps when it lands.

Wraps any ``Retriever`` (HybridRetriever, SemanticRetriever, etc.) and
mirrors its interface; transparent to the Pipeline.
"""
from orchestration.pipeline.contracts import EmbeddedQuery, RetrievalHit, MemoryItem
from orchestration.retriever.base import Retriever
from orchestration.sdp.entity_index import EntityIndex


DEFAULT_BOOST = 0.15


class EntityBoostedRetriever(Retriever):
    """Retriever wrapper that boosts entity-matched candidates."""

    def __init__(
        self,
        base: Retriever,
        index: EntityIndex | None = None,
        boost: float = DEFAULT_BOOST,
    ) -> None:
        self.base = base
        # Don't use `index or EntityIndex()` — an empty EntityIndex has
        # __len__ == 0 and is falsy, so a passed-in empty index would be
        # silently dropped and the caller's reference to it diverges.
        self.index = index if index is not None else EntityIndex()
        self.boost = boost

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
        self.index.add_many(items)

    # ----------------------------------------------------------------------

    def _maybe_boost(
        self, hits: list[RetrievalHit], query_text: str
    ) -> list[RetrievalHit]:
        if not hits:
            return hits
        matching_ids = self.index.lookup_query(query_text)
        if not matching_ids:
            return hits  # no entity hit in query — no boost

        boosted: list[RetrievalHit] = []
        for h in hits:
            if h.record.id in matching_ids:
                new_sim = min(1.0, h.similarity + self.boost)
                boosted.append(RetrievalHit(record=h.record, similarity=new_sim))
            else:
                boosted.append(h)
        boosted.sort(key=lambda x: -x.similarity)
        return boosted
