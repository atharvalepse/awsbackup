"""Ollama-backed Embedder using a local embedding model.

Hits the Ollama daemon's ``/api/embeddings`` endpoint. Default model is
``nomic-embed-text`` (768-dim); pull it once via ``ollama pull nomic-embed-text``.
Same pattern as :class:`HTTPOllamaClient` for SAM, but a separate small
client so embedder/ and sam/ stay decoupled.

Raises :class:`EmbedderError` when the daemon is unreachable, the model
isn't pulled, or the response has no embedding — callers can choose to
fall back to a stub.
"""
import httpx

from orchestration.embedder.base import Embedder
from orchestration.errors import EmbedderError
from orchestration.pipeline.contracts import Classification, EmbeddedQuery, Query


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT_SECONDS = 30.0


class OllamaEmbedder(Embedder):
    """Local semantic embedder via the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def version(self) -> str:
        return f"ollama:{self.model}"

    async def embed(
        self, query: Query, classification: Classification
    ) -> EmbeddedQuery:
        signal = query.text
        if classification.entities:
            signal += " || " + " ".join(sorted(classification.entities))

        body = {"model": self.model, "prompt": signal}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(f"{self.base_url}/api/embeddings", json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise EmbedderError(
                f"Ollama embed call failed: {type(exc).__name__}: {exc}"
            ) from exc

        embedding = data.get("embedding")
        if not embedding:
            raise EmbedderError(
                f"Ollama returned no embedding for model {self.model!r}; "
                "is the model pulled?"
            )
        return EmbeddedQuery(
            query=query,
            classification=classification,
            vector=list(embedding),
            embedder_version=self.version,
        )
