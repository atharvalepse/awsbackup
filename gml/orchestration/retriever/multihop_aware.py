"""MultiHopAwareRetriever — gate sliding-window memories by query type.

Sliding-window chunks (3-message overlaps) add useful cross-message
context for multi-hop questions but DILUTE single-hop retrieval — every
window competes with the individual messages it overlaps, so cross-encoder
candidate pools get noisier without proportional benefit.

This wrapper does the right thing per query type:
  - When classify_query.is_multi_hop is True: passes all candidates through
    (windows compete on merits — they often have the answer).
  - Otherwise: filters out candidates with source=="locomo-window" so
    single-hop questions get a clean candidate pool of single-message
    memories.

Net: windows live in the index (cheap), but only contribute to scoring
when they could actually help.

Composes around any base Retriever. Mirror the interface.
"""
import os

from orchestration.pipeline.contracts import EmbeddedQuery, MemoryItem, RetrievalHit
from orchestration.retriever.base import Retriever
from orchestration.sdp.query_router import classify_query


# Sources that are considered "multi-hop helper" memories — gated unless
# the query routes as multi-hop or list-style.
#
# - locomo-window: sliding-window message chunks
# - aal-entity-synth: per-entity aggregated snippets (Tier 3.2). These
#   collect every snippet about a single entity into one chunk. Great
#   for "What hobbies does Sam have?" (all 5 items in one retrieval),
#   noisy for "What car does Evan drive?" (mixes the car answer with
#   unrelated Evan snippets, dilutes cat-1 F1).
MULTI_HOP_ONLY_SOURCES = frozenset({"locomo-window", "aal-entity-synth"})


class MultiHopAwareRetriever(Retriever):
    """Wrap a retriever so sliding-window candidates only count when needed."""

    def __init__(
        self,
        base: Retriever,
        gated_sources: frozenset[str] | None = None,
    ) -> None:
        self.base = base
        self.gated_sources = gated_sources or MULTI_HOP_ONLY_SOURCES

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        hits = await self.base.search(embedded)
        return self._maybe_filter(hits, embedded.query.text)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        # Get more candidates up front so we have enough after filtering
        oversample = k * 2 if self.gated_sources else k
        hits = await self.base.get_top_matches(embedded, k=oversample)
        filtered = self._maybe_filter(hits, embedded.query.text)
        return filtered[:k]

    async def ingest(self, items: list[MemoryItem]) -> None:
        if hasattr(self.base, "ingest"):
            await self.base.ingest(items)

    def _maybe_filter(
        self, hits: list[RetrievalHit], query_text: str
    ) -> list[RetrievalHit]:
        if not hits or not self.gated_sources:
            return hits
        hints = classify_query(query_text)
        # Multi-hop / list-style questions BENEFIT from windows + entity
        # synths (they need to combine multiple facts). For everything
        # else, filter them out so the cross-encoder candidate pool is
        # clean and individual-message memories compete fairly.
        if hints.is_multi_hop:
            return hits
        # Cat-1 single-hop questions that LOOK like list questions also
        # benefit ("What kinds of activities does Sam do?" — classifier
        # tags this cat-1, but gold expects a list).
        try:
            from orchestration.sam.answer_generator import _is_list_question
            if _is_list_question(query_text):
                return hits
        except ImportError:
            pass
        # Pure single-hop: filter out the gated sources
        return [h for h in hits if h.record.source not in self.gated_sources]
