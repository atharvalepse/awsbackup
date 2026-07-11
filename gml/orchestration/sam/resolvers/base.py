from abc import ABC, abstractmethod

from orchestration.pipeline.contracts import MemoryItem


class ConflictResolver(ABC):
    """Internal SAM strategy. Returns mapping of item.id -> penalty in [0.0, 1.0]
    and a list of (loser_id, winner_id) supersession pairs.

    Higher penalty = stronger conflict signal. SAM uses penalty to drop items.
    """

    @abstractmethod
    async def score_conflicts(
        self, candidates: list[MemoryItem]
    ) -> tuple[dict[str, float], list[tuple[str, str]]]: ...
