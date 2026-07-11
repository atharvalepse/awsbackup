"""No-op ConflictResolver — useful when conflict resolution is disabled."""
from orchestration.pipeline.contracts import MemoryItem
from orchestration.sam.resolvers.base import ConflictResolver


class StubConflictResolver(ConflictResolver):
    async def score_conflicts(
        self, candidates: list[MemoryItem]
    ) -> tuple[dict[str, float], list[tuple[str, str]]]:
        return {}, []
