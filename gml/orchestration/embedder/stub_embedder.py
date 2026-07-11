"""Deterministic hash-derived Embedder.

Produces a stable, normalized vector from a SHA-256 hash of the input text.
Same text always yields the same vector. The vector is L2-normalized so
cosine similarity is just a dot product.

This is intentionally NOT a semantic embedding — it's a placeholder for
plumbing tests and for environments without an embedding API key. Real
semantic retrieval requires swapping in a true embedding model.
"""
from orchestration.embedder.base import Embedder
from orchestration.pipeline._stub_vectors import hash_to_unit_vector
from orchestration.pipeline.contracts import Classification, EmbeddedQuery, Query


DEFAULT_DIM = 384


class StubEmbedder(Embedder):
    """Hash-derived deterministic Embedder. No external calls."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    @property
    def version(self) -> str:
        return f"stub-sha256:dim={self.dim}"

    async def embed(
        self, query: Query, classification: Classification
    ) -> EmbeddedQuery:
        # Mix entities into the text so two queries with the same wording but
        # different extracted entities embed to different vectors.
        signal = query.text
        if classification.entities:
            signal += " || " + " ".join(sorted(classification.entities))

        vector = hash_to_unit_vector(signal, self.dim)
        return EmbeddedQuery(
            query=query,
            classification=classification,
            vector=vector,
            embedder_version=self.version,
        )
