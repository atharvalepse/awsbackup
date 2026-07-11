"""Tests for the BM25 lexical retriever."""
from datetime import datetime, timezone

import pytest

from orchestration.embedder import StubEmbedder
from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    MemoryItem,
)
from orchestration.retriever import BM25Retriever

from tests.conftest import make_query


def _item(id: str, content: str, entity: str | None = None) -> MemoryItem:
    return MemoryItem(
        id=id, content=content, entity=entity,
        timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.5,
    )


def _classification(entities=None):
    return Classification(
        intent_type="other",
        entities=entities or [],
        retrieval_hints={},
        confidence=0.5,
        source=ClassificationSource.KEYWORD_FALLBACK,
    )


@pytest.mark.asyncio
async def test_bm25_finds_exact_term(gpt_target):
    retriever = BM25Retriever([
        _item("a", "auth_service uses FastAPI"),
        _item("b", "payment_service uses PostgreSQL"),
        _item("c", "team standup is on Mondays"),
    ])
    embedded = await StubEmbedder().embed(
        make_query("FastAPI", gpt_target), _classification()
    )
    hits = await retriever.get_top_matches(embedded, k=3)
    assert hits, "BM25 should find the FastAPI mention"
    assert hits[0].record.id == "a"


@pytest.mark.asyncio
async def test_bm25_ignores_stub_vector(gpt_target):
    """BM25 search uses query text, not the EmbeddedQuery vector — so any
    vector should give the same lexical ranking.

    Needs >=3 docs so BM25Okapi's idf for the matching term is non-zero
    (with 2 docs, a 50% df gives idf=0)."""
    retriever = BM25Retriever([
        _item("a", "auth_service framework FastAPI"),
        _item("b", "completely unrelated content"),
        _item("c", "another unrelated doc about cats"),
        _item("d", "more unrelated text"),
    ])
    q = make_query("FastAPI", gpt_target)
    embedded = await StubEmbedder(dim=128).embed(q, _classification())
    hits = await retriever.get_top_matches(embedded, k=4)
    assert hits[0].record.id == "a"


@pytest.mark.asyncio
async def test_bm25_empty_corpus(gpt_target):
    retriever = BM25Retriever()
    embedded = await StubEmbedder().embed(make_query("anything", gpt_target), _classification())
    assert await retriever.search(embedded) == []


@pytest.mark.asyncio
async def test_bm25_no_matching_tokens(gpt_target):
    retriever = BM25Retriever([_item("a", "auth_service uses FastAPI")])
    embedded = await StubEmbedder().embed(
        make_query("hippopotamus", gpt_target), _classification()
    )
    hits = await retriever.search(embedded)
    assert hits == []
