"""SentenceTransformerEmbedder — drop-in Embedder backed by sentence-transformers.

Why this exists alongside FastEmbedEmbedder: fastembed only loads from its
curated ONNX list and can't read a local sentence-transformers directory.
Our FT'd embedder (models/embedder_locomo_ft) is saved by sentence-transformers'
`model.save()`, so this class is the loader that can pick it up.

Same dim as the base model (e.g. 384 for bge-small-en-v1.5) so it's a drop-in
replacement at the retriever level — no re-indexing needed if the base matches
what was previously used.

Usage:
    SentenceTransformerEmbedder(model_name="models/embedder_locomo_ft", device="mps")
"""
import asyncio
import os
from typing import Iterable

from orchestration.embedder.base import Embedder
from orchestration.errors import EmbedderError
from orchestration.pipeline.contracts import Classification, EmbeddedQuery, Query


DEFAULT_ST_EMBED_MODEL = os.environ.get(
    "GML_ST_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
)


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by sentence-transformers (loads local paths)."""

    def __init__(
        self,
        model_name: str = DEFAULT_ST_EMBED_MODEL,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "SentenceTransformerEmbedder requires sentence-transformers + torch. "
                "Install: pip install torch sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.device = device
        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._model.max_seq_length = max_length
        except Exception as exc:
            raise EmbedderError(
                f"SentenceTransformerEmbedder init failed for {model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @property
    def version(self) -> str:
        return f"st:{self.model_name}"

    async def embed(
        self, query: Query, classification: Classification
    ) -> EmbeddedQuery:
        signal = query.text
        if classification.entities:
            signal += " || " + " ".join(sorted(classification.entities))
        vector = await asyncio.to_thread(self._embed_one_sync, signal)
        return EmbeddedQuery(
            query=query,
            classification=classification,
            vector=vector,
            embedder_version=self.version,
        )

    def _embed_one_sync(self, text: str) -> list[float]:
        try:
            arr = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
            return [float(x) for x in arr]
        except Exception as exc:
            raise EmbedderError(
                f"SentenceTransformer embed failed: {type(exc).__name__}: {exc}"
            ) from exc

    async def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        return await asyncio.to_thread(self._embed_batch_sync, text_list)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            arrs = self._model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
            )
            return [[float(x) for x in arr] for arr in arrs]
        except Exception as exc:
            raise EmbedderError(
                f"SentenceTransformer batch embed failed: {type(exc).__name__}: {exc}"
            ) from exc
