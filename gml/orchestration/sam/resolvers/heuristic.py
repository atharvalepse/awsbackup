"""Entity-attribute disagreement resolver — penalizes older conflicting records."""
import logging

from orchestration.pipeline.contracts import MemoryItem
from orchestration.sam.resolvers.base import ConflictResolver


logger = logging.getLogger(__name__)

DEFAULT_MAX_PAIRS = 1000
_FULL_PENALTY_AGE_DAYS = 90.0


class HeuristicConflictResolver(ConflictResolver):
    """Detect conflicts via entity/attribute grouping; penalize older values.

    Algorithm:

    1. Group candidates by ``(entity, attribute)`` where both are non-None.
    2. Within each group, compare every pair. Values are normalized via
       ``.strip().lower()`` before comparison.
    3. For pairs with different values, the OLDER item receives a penalty
       proportional to age difference: ``min(1.0, age_diff_days / 90.0)``.
       Records the loser→winner pair for SAM's notes.
    4. Iteration is bounded by ``max_pairs``.
    """

    def __init__(self, max_pairs: int = DEFAULT_MAX_PAIRS) -> None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        self.max_pairs = max_pairs

    async def score_conflicts(
        self, candidates: list[MemoryItem]
    ) -> tuple[dict[str, float], list[tuple[str, str]]]:
        groups: dict[tuple[str, str], list[MemoryItem]] = {}
        for item in candidates:
            if item.entity and item.attribute:
                groups.setdefault((item.entity, item.attribute), []).append(item)

        penalties: dict[str, float] = {}
        supersessions: list[tuple[str, str]] = []
        pairs_examined = 0

        for group in groups.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if pairs_examined >= self.max_pairs:
                        logger.warning(
                            "HeuristicConflictResolver: max_pairs=%d reached",
                            self.max_pairs,
                            extra={"degraded_mode": True},
                        )
                        return penalties, supersessions
                    pairs_examined += 1

                    a, b = group[i], group[j]
                    if a.timestamp == b.timestamp:
                        continue

                    va = (a.value or "").strip().lower()
                    vb = (b.value or "").strip().lower()
                    if va == vb:
                        continue

                    if a.timestamp < b.timestamp:
                        older, newer = a, b
                    else:
                        older, newer = b, a

                    age_diff_days = (
                        newer.timestamp - older.timestamp
                    ).total_seconds() / 86400.0
                    penalty = min(1.0, age_diff_days / _FULL_PENALTY_AGE_DAYS)

                    existing = penalties.get(older.id, 0.0)
                    if penalty > existing:
                        penalties[older.id] = penalty
                        supersessions.append((older.id, newer.id))

        return penalties, supersessions
