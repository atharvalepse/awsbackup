from abc import ABC, abstractmethod

from orchestration.pipeline.contracts import Classification, Query


class Classifier(ABC):
    """Stage 1: classify the user's Query into a structured Classification.

    The Classifier returns intent type, entities, and retrieval hints. It is
    not responsible for any other pipeline concern — embedding, retrieval,
    pleasantry short-circuiting, etc. all live elsewhere.
    """

    @abstractmethod
    async def classify(self, query: Query) -> Classification: ...
