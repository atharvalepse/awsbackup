import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from typing import Iterable

from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    EmbeddedQuery,
    InterfaceType,
    ModelFamily,
    Query,
    TargetDescriptor,
)


# Throwaway target/classification used only to satisfy ``Query``'s schema when
# embedding raw document text via the default ``embed_batch`` below. Document
# embedding has no downstream model and no extracted entities, so only
# ``query.text`` ends up mattering — every embedder ignores the rest here.
_DOC_EMBED_TARGET = TargetDescriptor(
    model_family=ModelFamily.GPT,
    model_version="doc-embed",
    context_window=8192,
    interface_type=InterfaceType.OTHER,
)
_DOC_EMBED_CLASSIFICATION = Classification(
    intent_type="document",
    confidence=1.0,
    source=ClassificationSource.FAST_PATH,
)


class Embedder(ABC):
    """Stage 2: turn a Query (plus its Classification) into a dense vector.

    ``version`` ties an embedding to a specific embedder + model so callers
    (retrievers) can detect dimension mismatches against the vector store.
    """

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    async def embed(
        self, query: Query, classification: Classification
    ) -> EmbeddedQuery: ...

    async def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed raw document texts → one vector per text.

        This is the document-side counterpart to :meth:`embed` (which is
        query-shaped). It feeds the pgvector ``embedding`` column on the write
        path (:class:`PostgresMemoryStore`) and the JSONL→Postgres migration.

        The default implementation issues one :meth:`embed` call per text via
        a synthetic document query, with bounded concurrency
        (GML_EMBED_BATCH_CONCURRENCY, default 8) — sequential awaits made a
        1000-document ingest through a ~500ms-per-call network embedder take
        minutes instead of seconds. Batch-capable embedders (FastEmbed,
        SentenceTransformer) override this with a single vectorized pass.
        Result order matches input order regardless of completion order.
        """
        text_list = list(texts)
        if not text_list:
            return []
        limit = max(1, int(os.environ.get("GML_EMBED_BATCH_CONCURRENCY", "8")))
        sem = asyncio.Semaphore(limit)

        async def _one(text: str) -> list[float]:
            async with sem:
                query = Query(
                    text=text or "",
                    target=_DOC_EMBED_TARGET,
                    trace_id=uuid.uuid4().hex,
                )
                embedded = await self.embed(query, _DOC_EMBED_CLASSIFICATION)
                return embedded.vector

        return list(await asyncio.gather(*[_one(t) for t in text_list]))
