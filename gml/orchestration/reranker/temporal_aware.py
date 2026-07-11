"""TemporalAwareReranker — POST-rerank boost for date-bearing memories on
temporal queries.

Why this exists alongside ``TimeAwareRetriever``: that one boosts similarity
BEFORE the cross-encoder reranks. As the bench notes:

    "the date boost is small relative to cross-encoder's [-10,+10] reranking
    power"  — benchmark_locomo.py:438

i.e. a pre-rerank similarity boost of +0.15 gets clobbered by the CE's
re-scoring. The fix is to apply the boost AFTER the CE has produced its
final_score, so the lift survives all the way to retrieval output.

Heuristic:
  1. Run the base reranker as usual.
  2. If the query is classified as TEMPORAL (``QueryHints.is_temporal``):
     - For each ranked hit whose memory content matches a date pattern,
       boost ``final_score`` by ``boost`` (default 0.4 — large enough to
       actually re-order top-K, capped at 1.0).
     - Re-sort by adjusted final_score.
  3. For non-temporal queries: pure pass-through.

Configuration:
  GML_TEMPORAL_BOOST   : magnitude of the post-rerank boost (default 0.4).
                         Disable by setting to 0.

Composes around any Reranker — designed to wrap the existing
TwoStage/Ensemble + NegationAware chain.
"""
import os

from orchestration.pipeline.contracts import Query, RankedHit, RetrievalHit
from orchestration.reranker.base import Reranker
from orchestration.sdp.date_extractor import has_date
from orchestration.sdp.query_router import classify_query


DEFAULT_TEMPORAL_BOOST = float(os.environ.get("GML_TEMPORAL_BOOST", "0.4"))


class TemporalAwareReranker(Reranker):
    """Boost final_score of date-bearing hits on temporal queries (post-CE)."""

    def __init__(
        self,
        base: Reranker,
        boost: float = DEFAULT_TEMPORAL_BOOST,
    ) -> None:
        self.base = base
        self.boost = boost

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        # Pull a wider set from the base so we have date-bearing hits to
        # promote even if they were ranked below top-k pre-temporal.
        wide_k = max(k * 3, 30) if self.boost > 0 else k
        ranked = await self.base.pick_best(hits, query, k=wide_k)
        if not ranked or self.boost <= 0:
            return ranked[:k]

        hints = classify_query(query.text)
        if not hints.is_temporal:
            return ranked[:k]

        # Boost date-bearing candidates so they can climb past the CE top.
        adjusted: list[RankedHit] = []
        for rh in ranked:
            if has_date(rh.hit.record.content):
                new_score = min(1.0, rh.final_score + self.boost)
                note = f" +temporal({self.boost:.2f})"
                adjusted.append(rh.model_copy(update={
                    "final_score": new_score,
                    "score_reason": rh.score_reason + note,
                }))
            else:
                adjusted.append(rh)
        adjusted.sort(key=lambda r: r.final_score, reverse=True)
        return adjusted[:k]
