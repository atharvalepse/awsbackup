"""Strong local embeddings via the `fastembed` library (ONNX-based).

Default model: ``BAAI/bge-small-en-v1.5`` — 384-dim, top of MTEB at its
size class, runs ~20x faster than sentence-transformers via ONNX runtime.
Swap to ``BAAI/bge-large-en-v1.5`` (1024-dim) or ``intfloat/e5-large-v2``
for higher LOCOMO scores at the cost of more RAM and slower inference.

First use downloads the ONNX weights (~130 MB for bge-small, ~1.3 GB for
bge-large) and caches them under ``~/.cache/fastembed``.
"""
import asyncio
import os
from typing import Iterable

from orchestration.embedder.base import Embedder
from orchestration.errors import EmbedderError
from orchestration.pipeline.contracts import Classification, EmbeddedQuery, Query


# Strong defaults tuned for the LOCOMO-style benchmarks.
SMALL_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ~130MB  (fastest)
LARGE_MODEL = "BAAI/bge-large-en-v1.5"   # 1024-dim, ~1.3GB (best retrieval)
MULTILINGUAL = "intfloat/multilingual-e5-large"  # 1024-dim, ~2GB

# Override at runtime with env var GML_EMBED_MODEL — any fastembed-supported
# model id. Defaults to bge-small for fast iteration; switch to bge-large
# for production / benchmark runs (~3-5% absolute LOCOMO recall gain).
DEFAULT_MODEL = os.environ.get("GML_EMBED_MODEL", SMALL_MODEL)


class FastEmbedEmbedder(Embedder):
    """ONNX-based strong-semantic Embedder.

    Construction loads the model (downloading on first use). Use one
    instance per process and share it across the pipeline + retriever.

    Args:
        model_name: any fastembed-supported model id.
        cache_dir: where to keep ONNX weights. Defaults to fastembed's
            built-in ``~/.cache/fastembed``.
        max_length: tokenizer truncation length for the model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                max_length=max_length,
            )
        except Exception as exc:
            raise EmbedderError(
                f"FastEmbedEmbedder init failed for {model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @property
    def version(self) -> str:
        return f"fastembed:{self.model_name}"

    async def embed(
        self, query: Query, classification: Classification
    ) -> EmbeddedQuery:
        signal = query.text
        if classification.entities:
            signal += " || " + " ".join(sorted(classification.entities))

        # fastembed is synchronous + CPU-bound. Offload to a thread so the
        # event loop stays responsive when multiple embeddings run.
        vector = await asyncio.to_thread(self._embed_one_sync, signal)
        return EmbeddedQuery(
            query=query,
            classification=classification,
            vector=vector,
            embedder_version=self.version,
        )

    def _embed_one_sync(self, text: str) -> list[float]:
        try:
            it = self._model.embed([text])
            arr = next(iter(it))
            return [float(x) for x in arr]
        except Exception as exc:
            raise EmbedderError(
                f"FastEmbed embed failed: {type(exc).__name__}: {exc}"
            ) from exc

    async def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Batch embedding for record ingestion. Much faster than calling
        ``embed`` in a loop because fastembed amortizes ONNX overhead."""
        text_list = list(texts)
        if not text_list:
            return []
        return await asyncio.to_thread(self._embed_batch_sync, text_list)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            return [
                [float(x) for x in arr]
                for arr in self._model.embed(texts)
            ]
        except Exception as exc:
            raise EmbedderError(
                f"FastEmbed batch embed failed: {type(exc).__name__}: {exc}"
            ) from exc
