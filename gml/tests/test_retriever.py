import pytest

from orchestration.embedder import StubEmbedder
from orchestration.pipeline.contracts import Classification, ClassificationSource
from orchestration.retriever import StubRetriever, default_records

from tests.conftest import make_query


@pytest.fixture
def classification():
    return Classification(
        intent_type="other",
        entities=[],
        retrieval_hints={},
        confidence=0.5,
        source=ClassificationSource.KEYWORD_FALLBACK,
    )


@pytest.mark.asyncio
async def test_default_records_has_eight_with_conflict_pair():
    records = default_records()
    assert len(records) == 8
    ids = {r.id for r in records}
    assert {"m-1", "m-2"}.issubset(ids)
    m1 = next(r for r in records if r.id == "m-1")
    assert m1.pinned is True


@pytest.mark.asyncio
async def test_search_returns_some_hits(gpt_target, classification):
    embedder = StubEmbedder(dim=384)
    retriever = StubRetriever(dim=384)
    embedded = await embedder.embed(make_query("auth_service framework", gpt_target), classification)
    hits = await retriever.search(embedded)
    assert len(hits) > 0
    # Sorted descending
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.asyncio
async def test_get_top_matches_respects_k(gpt_target, classification):
    embedder = StubEmbedder(dim=384)
    retriever = StubRetriever(dim=384)
    embedded = await embedder.embed(make_query("any query", gpt_target), classification)
    top3 = await retriever.get_top_matches(embedded, k=3)
    assert len(top3) <= 3


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_records_pass_threshold(gpt_target, classification):
    embedder = StubEmbedder(dim=384)
    # threshold above 1.0 → nothing can match
    retriever = StubRetriever(dim=384, match_threshold=2.0)
    embedded = await embedder.embed(make_query("anything", gpt_target), classification)
    assert await retriever.search(embedded) == []
