from abc import ABC, abstractmethod

from orchestration.pipeline.contracts import Query, RankedHit, RetrievalHit


class Reranker(ABC):
    """Stage 4: re-rank a wide set of retrieval hits down to the best ``k``.

    The Reranker is PURE compute. Conflict resolution is NOT a Reranker
    concern — that lives in SAM, called after this stage.
    """

    @abstractmethod
    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]: ...
