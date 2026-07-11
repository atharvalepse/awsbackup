from abc import ABC, abstractmethod

from orchestration.pipeline.contracts import EmbeddedQuery, RetrievalHit


class Retriever(ABC):
    """Stage 3: vector search over a memory store.

    Two distinct entry points mirror the pipeline's "found anything?" branch:

    - ``search``: cheap probe — does anything semantically relevant exist?
      Returns whatever the store finds above an internal threshold, possibly
      empty. Pipeline uses the result to choose between the YES and NO paths.

    - ``get_top_matches``: full top-k retrieval — only called when ``search``
      returned non-empty. Returns up to ``k`` hits sorted by similarity desc.
    """

    @abstractmethod
    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]: ...

    @abstractmethod
    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]: ...
