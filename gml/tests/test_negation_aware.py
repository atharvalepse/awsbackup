"""Tests for orchestration/reranker/negation_aware.py — polarity scoring."""
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import (
    MemoryItem,
    Query,
    RankedHit,
    RetrievalHit,
    TargetDescriptor,
)
from orchestration.reranker.base import Reranker
from orchestration.reranker.negation_aware import (
    NegationAwareReranker,
    is_negated_query,
    memory_is_negated,
)


def _record(rid: str, content: str, aal_negated: bool | None = None) -> MemoryItem:
    raw = {}
    if aal_negated is not None:
        raw["tuple"] = {"subject": "x", "verb": "y", "object": "z", "negated": aal_negated}
    return MemoryItem(
        id=rid, content=content, timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.7, pinned=False,
        raw_metadata=raw,
    )


def _ranked(rec_id: str, content: str, score: float, aal_negated=None) -> RankedHit:
    rec = _record(rec_id, content, aal_negated)
    hit = RetrievalHit(record=rec, similarity=score)
    return RankedHit(
        hit=hit, semantic_score=score, recency_score=1.0,
        authority_score=0.7, pin_boost=0.0, final_score=score, score_reason="",
    )


def _query(text: str) -> Query:
    return Query(text=text, target=TargetDescriptor.for_claude(),
                 session_context={}, trace_id="t")


class _FakeReranker(Reranker):
    """Returns ranked input unchanged."""
    def __init__(self, ranked: list[RankedHit]) -> None:
        self._ranked = ranked

    async def pick_best(self, hits, query, k=10):
        return list(self._ranked)[:k]


class TestNegationDetection:
    def test_query_positive(self):
        assert is_negated_query("do we use Redis?") is False

    def test_query_negative(self):
        assert is_negated_query("do we NOT use Redis anymore?") is True
        assert is_negated_query("we don't use it") is True
        assert is_negated_query("we never used Postgres") is True

    def test_query_empty(self):
        assert is_negated_query("") is False

    def test_aal_tuple_flag_trusted(self):
        # Even if content has "use", the structured flag wins
        rec = _record("r1", "we use Redis", aal_negated=True)
        assert memory_is_negated(rec) is True

    def test_aal_tuple_positive(self):
        rec = _record("r1", "never used it", aal_negated=False)
        # Tuple wins over content
        assert memory_is_negated(rec) is False

    def test_no_tuple_uses_content(self):
        rec = _record("r1", "we don't use Redis")
        assert memory_is_negated(rec) is True

    def test_no_tuple_positive_content(self):
        rec = _record("r1", "Redis is our session cache")
        assert memory_is_negated(rec) is False


@pytest.mark.asyncio
class TestNegationAwareReranker:
    async def test_agreement_boosts(self):
        """Positive query + positive memory → +5% boost."""
        base = _FakeReranker([_ranked("a", "we use Redis", 0.8)])
        rr = NegationAwareReranker(base)
        out = await rr.pick_best([], _query("do we use Redis?"), k=1)
        assert out[0].final_score > 0.8
        assert "polarity_agree" in out[0].score_reason

    async def test_disagreement_demotes(self):
        """Positive query + negated memory → 0.7× demotion."""
        base = _FakeReranker([_ranked("a", "we use Redis", 0.8, aal_negated=True)])
        rr = NegationAwareReranker(base)
        out = await rr.pick_best([], _query("do we use Redis?"), k=1)
        assert out[0].final_score == pytest.approx(0.8 * 0.7)
        assert "polarity_demote" in out[0].score_reason

    async def test_reorders_by_polarity(self):
        """When polarities disagree, demote until correct-polarity wins."""
        # Original: negated memory is at 0.85, positive memory at 0.7
        # After negation-aware: negated×0.7=0.595, positive×1.05=0.735
        # → positive wins
        base = _FakeReranker([
            _ranked("a", "we don't use Redis", 0.85, aal_negated=True),
            _ranked("b", "we use Redis", 0.7, aal_negated=False),
        ])
        rr = NegationAwareReranker(base)
        out = await rr.pick_best([], _query("do we use Redis?"), k=2)
        assert out[0].hit.record.id == "b"

    async def test_empty_input(self):
        rr = NegationAwareReranker(_FakeReranker([]))
        out = await rr.pick_best([], _query("anything"), k=5)
        assert out == []

    async def test_negative_query_finds_negated_memory(self):
        """Negative query + negated memory → agree, boost."""
        base = _FakeReranker([_ranked("a", "we don't use Redis", 0.8, aal_negated=True)])
        rr = NegationAwareReranker(base)
        out = await rr.pick_best([], _query("we don't use Redis right?"), k=1)
        assert "polarity_agree" in out[0].score_reason
