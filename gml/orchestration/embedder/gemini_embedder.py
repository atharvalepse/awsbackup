"""Gemini-backed Embedder using google-genai's embedding API.

Default model is ``text-embedding-004`` (768-dim). Gated on ``GEMINI_API_KEY``
— when unset, construction succeeds but every ``embed`` call raises
:class:`EmbedderError` so the caller can fall back to a stub Embedder.
"""
import os

from orchestration.embedder.base import Embedder
from orchestration.errors import EmbedderError
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import Classification, EmbeddedQuery, Query


slog = StructuredLogger("embedder.gemini")

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIM = 3072


class GeminiEmbedder(Embedder):
    """Real semantic embedder via Gemini text-embedding-004."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.task_type = task_type

    @property
    def version(self) -> str:
        return f"gemini:{self.model}"

    async def embed(
        self, query: Query, classification: Classification
    ) -> EmbeddedQuery:
        if self.api_key is None:
            raise EmbedderError(
                "GeminiEmbedder requires GEMINI_API_KEY; none was provided"
            )

        # Mix entities into the embed signal so two queries with same
        # wording but different extracted entities get different vectors.
        signal = query.text
        if classification.entities:
            signal += " || " + " ".join(sorted(classification.entities))

        try:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=self.api_key)
            config = genai_types.EmbedContentConfig(task_type=self.task_type)
            response = await client.aio.models.embed_content(
                model=self.model,
                contents=[signal],
                config=config,
            )
        except Exception as exc:
            raise EmbedderError(
                f"Gemini embed call failed: {type(exc).__name__}: {exc}"
            ) from exc

        # The SDK returns response.embeddings — a list of ContentEmbedding
        # objects with a .values list[float]. Defensive extraction:
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            raise EmbedderError("Gemini embed returned no embeddings")
        first = embeddings[0]
        values = getattr(first, "values", None)
        if not values:
            raise EmbedderError("Gemini embed returned an empty embedding")

        return EmbeddedQuery(
            query=query,
            classification=classification,
            vector=list(values),
            embedder_version=self.version,
        )
