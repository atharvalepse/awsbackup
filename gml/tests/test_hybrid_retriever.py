"""Tests for the RRF-fused hybrid retriever (dense + BM25)."""
import math
from datetime import datetime, timezone

import pytest

from orchestration.embedder.base import Embedder
from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    EmbeddedQuery,
    MemoryItem,
    Query,
)
from orchestration.retriever import (
    BM25Retriever,
    HybridRetriever,
    SemanticRetriever,
)

from tests.conftest import make_query


class _DetEmbedder(Embedder):
    """Deterministic test embedder — vector is the byte values of the text
    padded to ``dim``, L2-normalized. Same text → same vector."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    @property
    def version(self) -> str:
        return f"det:{self.dim}"

    async def embed(self, query: Query, classification: Classification) -> EmbeddedQuery:
        text = query.text
        if classification.entities:
            text += " || " + " ".join(sorted(classification.entities))
        bytes_ = text.encode("utf-8")
        vec = [float(b) for b in bytes_[: self.dim]]
        while len(vec) < self.dim:
            vec.append(0.0)
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / n for x in vec]
        return EmbeddedQuery(query=query, classification=classification,
                             vector=vec, embedder_version=self.version)


def _records():
    now = datetime.now(timezone.utc)
    return [
        MemoryItem(
            id="auth", content="auth_service is implemented in FastAPI",
            entity="auth_service", attribute="framework", value="FastAPI",
            timestamp=now, source="adr", authority_score=0.9,
        ),
        MemoryItem(
            id="payments", content="payment_service uses PostgreSQL 16",
            entity="payment_service", attribute="database",
            timestamp=now, source="config", authority_score=0.7,
        ),
        MemoryItem(
            id="standup", content="standup is at 10am PT on Mondays",
            timestamp=now, source="handbook", authority_score=0.5,
        ),
    ]


@pytest.fixture
def cls():
    return Classification(
        intent_type="other", entities=[], retrieval_hints={},
        confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
    )


@pytest.mark.asyncio
async def test_hybrid_fuses_dense_and_sparse(gpt_target, cls):
    records = _records()
    embedder = _DetEmbedder(dim=32)
    dense = SemanticRetriever(embedder=embedder)
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    await hybrid.ingest(records)
    assert len(dense.records) == 3
    assert len(sparse.records) == 3

    embedded = await embedder.embed(make_query("FastAPI auth_service", gpt_target), cls)
    fused = await hybrid.get_top_matches(embedded, k=3)
    # 'auth' should win because BM25 sees "FastAPI" + "auth_service" exact match
    assert fused, "hybrid should return hits"
    assert fused[0].record.id == "auth"
    sims = [h.similarity for h in fused]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.asyncio
async def test_hybrid_handles_empty_dense_or_sparse(gpt_target, cls):
    # No records: both dense and sparse return empty → hybrid empty
    embedder = _DetEmbedder(dim=32)
    dense = SemanticRetriever(embedder=embedder)
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    embedded = await embedder.embed(make_query("query", gpt_target), cls)
    assert await hybrid.search(embedded) == []


@pytest.mark.asyncio
async def test_hybrid_rrf_normalized_to_unit(gpt_target, cls):
    embedder = _DetEmbedder(dim=32)
    dense = SemanticRetriever(embedder=embedder)
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    await hybrid.ingest(_records())
    embedded = await embedder.embed(make_query("FastAPI", gpt_target), cls)
    fused = await hybrid.search(embedded)
    assert all(0.0 <= h.similarity <= 1.0 for h in fused)
