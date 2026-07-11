"""Weighted-score Reranker — semantic + recency + authority + pin.

Pure compute, no conflict resolution. Takes RetrievalHits (which carry a
similarity score from the vector store) and produces RankedHits ordered by
``final_score`` descending, capped at ``k``.
"""
from datetime import datetime, timezone
from typing import Callable

from orchestration.errors import RerankerError
from orchestration.pipeline.contracts import OrchestrationConfig, Query, RankedHit, RetrievalHit
from orchestration.reranker.base import Reranker


_EXPECTED_WEIGHT_KEYS = frozenset({"semantic", "recency", "authority", "pin"})


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


class ScoreReranker(Reranker):
    """Weighted-sum Reranker. Semantic score == vector-store similarity.

    final_score = w_sem * similarity
                + w_rec * recency_decay
                + w_auth * authority_score
                + w_pin * (1 if pinned else 0)
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        missing = _EXPECTED_WEIGHT_KEYS - set(config.ranking_weights.keys())
        if missing:
            raise RerankerError(
                f"ranking_weights missing keys: {sorted(missing)}"
            )
        self.config = config
        self.now_provider = now_provider or _default_now

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        now = self.now_provider()
        weights = self.config.ranking_weights
        half_life = self.config.recency_half_life_days

        ranked: list[RankedHit] = []
        for hit in hits:
            record = hit.record
            semantic_score = hit.similarity
            recency_score = _recency(now, record.timestamp, half_life)
            authority_score = record.authority_score
            pin_boost = 1.0 if record.pinned else 0.0

            final_score = (
                weights["semantic"] * semantic_score
                + weights["recency"] * recency_score
                + weights["authority"] * authority_score
                + weights["pin"] * pin_boost
            )

            score_reason = (
                f"semantic={semantic_score:.2f}, "
                f"recency={recency_score:.2f}, "
                f"authority={authority_score:.2f}, "
                f"pin={pin_boost:.2f}, "
                f"final={final_score:.2f}"
            )

            ranked.append(
                RankedHit(
                    hit=hit,
                    semantic_score=semantic_score,
                    recency_score=recency_score,
                    authority_score=authority_score,
                    pin_boost=pin_boost,
                    final_score=final_score,
                    score_reason=score_reason,
                )
            )

        ranked.sort(key=lambda rh: rh.final_score, reverse=True)
        return ranked[:k]


def _recency(now: datetime, timestamp: datetime, half_life_days: float) -> float:
    age_days = (now - timestamp).total_seconds() / 86400.0
    return min(1.0, 0.5 ** (age_days / half_life_days))
