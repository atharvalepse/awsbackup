"""Tests for SemanticRetriever paired with mock + real embedders."""
import math

import pytest

from orchestration.embedder.base import Embedder
from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    EmbeddedQuery,
    Query,
)
from orchestration.retriever import SemanticRetriever, default_records

from tests.conftest import make_query


class _DeterministicEmbedder(Embedder):
    """Cheap repeatable Embedder for tests — vector is the byte values of the
    text, padded/truncated to ``dim``. Same text → same vector; different
    text → different vector."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    @property
    def version(self) -> str:
        return f"deterministic:dim={self.dim}"

    async def embed(self, query: Query, classification: Classification) -> EmbeddedQuery:
        text = query.text
        if classification.entities:
            text += " || " + " ".join(sorted(classification.entities))
        bytes_ = text.encode("utf-8")
        vec = [float(b) for b in bytes_[: self.dim]]
        while len(vec) < self.dim:
            vec.append(0.0)
        # L2-normalize so cosine == dot product.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        return EmbeddedQuery(
            query=query, classification=classification,
            vector=vec, embedder_version=self.version,
        )


def _classification():
    return Classification(
        intent_type="other", entities=[], retrieval_hints={},
        confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
    )


@pytest.mark.asyncio
async def test_semantic_retriever_ingest_then_search(gpt_target):
    embedder = _DeterministicEmbedder(dim=32)
    retriever = SemanticRetriever(embedder=embedder)
    await retriever.ingest(default_records())
    assert len(retriever.records) == 8

    embedded = await embedder.embed(
        make_query("auth_service framework", gpt_target), _classification()
    )
    hits = await retriever.search(embedded)
    assert len(hits) > 0
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.asyncio
async def test_semantic_retriever_query_record_alignment(gpt_target):
    """Same text on the query side and record side should produce similarity ≈ 1."""
    embedder = _DeterministicEmbedder(dim=32)
    retriever = SemanticRetriever(embedder=embedder)
    records = default_records()
    await retriever.ingest(records)

    target_record = records[0]
    embedded = await embedder.embed(
        make_query(target_record.content, gpt_target), _classification()
    )
    hits = await retriever.search(embedded)
    # The matching record's similarity should be highest (close to 1.0)
    top = hits[0]
    assert top.record.id == target_record.id
    assert top.similarity > 0.9


@pytest.mark.asyncio
async def test_semantic_retriever_get_top_matches_respects_k(gpt_target):
    embedder = _DeterministicEmbedder(dim=32)
    retriever = SemanticRetriever(embedder=embedder)
    await retriever.ingest(default_records())
    embedded = await embedder.embed(make_query("any query", gpt_target), _classification())
    top3 = await retriever.get_top_matches(embedded, k=3)
    assert len(top3) <= 3


@pytest.mark.asyncio
async def test_semantic_retriever_empty_without_ingest(gpt_target):
    embedder = _DeterministicEmbedder(dim=32)
    retriever = SemanticRetriever(embedder=embedder)
    embedded = await embedder.embed(make_query("anything", gpt_target), _classification())
    assert await retriever.search(embedded) == []
    assert await retriever.get_top_matches(embedded) == []
