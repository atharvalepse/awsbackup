"""Tests for orchestration/retriever/entity_boosted.py — entity boost wrapper."""
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import (
    EmbeddedQuery,
    MemoryItem,
    Query,
    RetrievalHit,
    TargetDescriptor,
)
from orchestration.pipeline.contracts import Classification, ClassificationSource
from orchestration.retriever.base import Retriever
from orchestration.retriever.entity_boosted import EntityBoostedRetriever
from orchestration.sdp.entity_index import EntityIndex


def _record(rec_id: str, content: str) -> MemoryItem:
    return MemoryItem(
        id=rec_id, content=content, timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.7, pinned=False,
    )


def _query(text: str) -> Query:
    return Query(text=text, target=TargetDescriptor.for_claude(),
                 session_context={}, trace_id="t")


def _embedded(text: str) -> EmbeddedQuery:
    return EmbeddedQuery(
        query=_query(text),
        classification=Classification(
            intent_type="question", entities=[], retrieval_hints={},
            confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
        ),
        vector=[0.0] * 384,
        embedder_version="stub",
    )


class _FakeRetriever(Retriever):
    """Returns a fixed list of hits regardless of query."""

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = list(hits)
        self.ingested: list[MemoryItem] = []

    async def search(self, embedded):
        return list(self._hits)

    async def get_top_matches(self, embedded, k=50):
        return list(self._hits)[:k]

    async def ingest(self, items):
        self.ingested.extend(items)


@pytest.mark.asyncio
async def test_no_entity_in_query_passes_through():
    """If query has no entity surface, retriever output is unchanged."""
    base = _FakeRetriever([
        RetrievalHit(record=_record("r1", "alpha"), similarity=0.6),
        RetrievalHit(record=_record("r2", "beta"), similarity=0.5),
    ])
    idx = EntityIndex()
    # Index has Caroline but the query doesn't mention her
    idx.add(_record("r1", "Caroline went home"))

    wrapped = EntityBoostedRetriever(base, idx)
    hits = await wrapped.get_top_matches(_embedded("what is the answer"))
    # Order should be unchanged — no boost applied
    assert [h.record.id for h in hits] == ["r1", "r2"]
    assert hits[0].similarity == 0.6  # unchanged


@pytest.mark.asyncio
async def test_entity_match_boosts_score_and_reorders():
    """Entity-matched candidates should jump to the top."""
    rec1 = _record("r1", "Caroline likes coffee")  # entity: Caroline
    rec2 = _record("r2", "the weather is nice")    # no Caroline

    base = _FakeRetriever([
        RetrievalHit(record=rec2, similarity=0.8),   # higher cosine
        RetrievalHit(record=rec1, similarity=0.6),
    ])
    idx = EntityIndex()
    idx.add(rec1)
    idx.add(rec2)

    wrapped = EntityBoostedRetriever(base, idx, boost=0.3)
    hits = await wrapped.get_top_matches(_embedded("what does Caroline like?"))
    # rec1 now has 0.6 + 0.3 = 0.9; rec2 stays at 0.8.
    assert hits[0].record.id == "r1"
    assert hits[0].similarity == pytest.approx(0.9)
    assert hits[1].record.id == "r2"
    assert hits[1].similarity == 0.8


@pytest.mark.asyncio
async def test_search_probe_also_boosts():
    """search() also applies the entity boost."""
    rec1 = _record("r1", "Caroline likes coffee")
    base = _FakeRetriever([
        RetrievalHit(record=_record("r2", "weather"), similarity=0.8),
        RetrievalHit(record=rec1, similarity=0.5),
    ])
    idx = EntityIndex()
    idx.add(rec1)
    wrapped = EntityBoostedRetriever(base, idx, boost=0.4)
    hits = await wrapped.search(_embedded("about Caroline"))
    assert hits[0].record.id == "r1"
    assert hits[0].similarity == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_ingest_updates_entity_index():
    """Ingest should populate the entity index automatically."""
    base = _FakeRetriever([])
    idx = EntityIndex()
    wrapped = EntityBoostedRetriever(base, idx)
    rec = _record("r1", "Caroline came over today")
    await wrapped.ingest([rec])
    assert idx.lookup_query("Caroline") == {"r1"}
    # Underlying base also ingests
    assert base.ingested == [rec]


@pytest.mark.asyncio
async def test_boost_cap_at_one():
    """Similarity can never exceed 1.0 after boost."""
    rec1 = _record("r1", "Caroline alone")
    base = _FakeRetriever([
        RetrievalHit(record=rec1, similarity=0.95),
    ])
    idx = EntityIndex()
    idx.add(rec1)
    wrapped = EntityBoostedRetriever(base, idx, boost=0.5)
    hits = await wrapped.get_top_matches(_embedded("Caroline?"))
    assert hits[0].similarity == 1.0  # clamped


@pytest.mark.asyncio
async def test_empty_hits():
    """Empty input → empty output, no crash."""
    base = _FakeRetriever([])
    wrapped = EntityBoostedRetriever(base, EntityIndex())
    assert await wrapped.search(_embedded("anything")) == []
    assert await wrapped.get_top_matches(_embedded("anything")) == []
