"""Tests for MultiHopAwareRetriever — source-based filtering by query type."""
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import (
    EmbeddedQuery, MemoryItem, Query, RetrievalHit, TargetDescriptor,
    Classification, ClassificationSource,
)
from orchestration.retriever.multihop_aware import (
    MULTI_HOP_ONLY_SOURCES,
    MultiHopAwareRetriever,
)


class _FakeRetriever:
    """Stub: returns a fixed set of hits regardless of query."""

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

    async def search(self, embedded):
        return list(self._hits)

    async def get_top_matches(self, embedded, k: int = 50):
        return list(self._hits[:k])

    async def ingest(self, items):
        return None


def _hit(mid: str, content: str, source: str, sim: float = 0.8) -> RetrievalHit:
    rec = MemoryItem(
        id=mid, content=content, source=source,
        timestamp=datetime.now(timezone.utc),
        authority_score=0.7, pinned=False,
    )
    return RetrievalHit(record=rec, similarity=sim)


def _eq(text: str) -> EmbeddedQuery:
    target = TargetDescriptor.for_claude()
    classification = Classification(
        intent_type="task", confidence=0.9,
        source=ClassificationSource.LLM, degraded=False,
    )
    q = Query(text=text, target=target, session_context={}, trace_id="t-1")
    return EmbeddedQuery(
        query=q, classification=classification,
        vector=[0.1] * 8, embedder_version="stub",
    )


@pytest.mark.asyncio
async def test_synths_pass_through_on_multi_hop():
    hits = [
        _hit("raw-1", "Caroline did yoga", "locomo-raw", 0.7),
        _hit("synth-1", "About Caroline:\n- yoga\n- meditation", "aal-entity-synth", 0.8),
        _hit("win-1", "Caroline: yoga ... meditation", "locomo-window", 0.6),
    ]
    rr = MultiHopAwareRetriever(_FakeRetriever(hits))
    # "all the X" triggers query_router.is_multi_hop
    out = await rr.search(_eq("What are all the practices Caroline does?"))
    sources = {h.record.source for h in out}
    assert "aal-entity-synth" in sources, "synth must survive on multi-hop"
    assert "locomo-window" in sources


@pytest.mark.asyncio
async def test_synths_filtered_on_single_hop_non_list():
    hits = [
        _hit("raw-1", "Evan drives a Prius", "locomo-raw", 0.9),
        _hit("synth-1", "About Evan:\n- drives Prius\n- visited Jasper", "aal-entity-synth", 0.85),
        _hit("win-1", "Evan: ... Prius ...", "locomo-window", 0.6),
    ]
    rr = MultiHopAwareRetriever(_FakeRetriever(hits))
    # Bare cat-1 question — synth would pollute precision
    out = await rr.search(_eq("What car does Evan drive?"))
    sources = {h.record.source for h in out}
    assert "aal-entity-synth" not in sources, "synth should be filtered on cat-1"
    assert "locomo-window" not in sources
    assert "locomo-raw" in sources


@pytest.mark.asyncio
async def test_synths_pass_through_on_list_question():
    """Cat-1 single-hop classifier tag but list-style gold —
    entity synth must survive (gold expects multiple items)."""
    hits = [
        _hit("raw-1", "Sam paints", "locomo-raw", 0.7),
        _hit("synth-1", "About Sam:\n- paints\n- hikes\n- runs", "aal-entity-synth", 0.85),
    ]
    rr = MultiHopAwareRetriever(_FakeRetriever(hits))
    out = await rr.search(_eq("What kinds of activities does Sam do?"))
    sources = {h.record.source for h in out}
    assert "aal-entity-synth" in sources


@pytest.mark.asyncio
async def test_window_filtered_on_single_hop_unchanged():
    """Pre-existing behaviour: window-source filtered when not multi-hop."""
    hits = [
        _hit("raw-1", "Mel is in Tokyo", "locomo-raw", 0.9),
        _hit("win-1", "Mel: ... Tokyo ...", "locomo-window", 0.7),
    ]
    rr = MultiHopAwareRetriever(_FakeRetriever(hits))
    out = await rr.search(_eq("Where is Mel?"))
    sources = {h.record.source for h in out}
    assert "locomo-window" not in sources


def test_default_gated_sources():
    assert "locomo-window" in MULTI_HOP_ONLY_SOURCES
    assert "aal-entity-synth" in MULTI_HOP_ONLY_SOURCES
