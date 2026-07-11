"""Tests for EnsembleCrossEncoder.

The ensemble averages two reranker scores; these tests use mock rerankers
so we don't need to load any real cross-encoder models.
"""
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import MemoryItem, RankedHit, RetrievalHit
from orchestration.reranker import EnsembleCrossEncoder
from orchestration.reranker.base import Reranker

from tests.conftest import make_query


def _item(id: str) -> RetrievalHit:
    rec = MemoryItem(
        id=id,
        content=f"content-{id}",
        timestamp=datetime.now(timezone.utc),
        source="test",
        authority_score=0.5,
    )
    return RetrievalHit(record=rec, similarity=0.5)


class _StubReranker(Reranker):
    """Returns RankedHits with caller-provided scores, by record id."""

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self.scores_by_id = scores_by_id

    async def pick_best(self, hits, query, k=10):
        out: list[RankedHit] = []
        ordered = sorted(
            hits, key=lambda h: -self.scores_by_id.get(h.record.id, 0.0)
        )
        for h in ordered[:k]:
            s = self.scores_by_id.get(h.record.id, 0.0)
            out.append(RankedHit(
                hit=h, semantic_score=s, recency_score=0.0,
                authority_score=0.0, pin_boost=0.0, final_score=s,
                score_reason=f"stub: {s}",
            ))
        return out


@pytest.mark.asyncio
async def test_ensemble_averages_scores(gpt_target):
    """Two rerankers disagree on top-1; the ensemble breaks the tie."""
    # Primary likes a (0.9), secondary likes b (0.9).
    # Equal-weight blend gives both 0.55, but c is consistent winner.
    primary = _StubReranker({"a": 0.9, "b": 0.2, "c": 0.7})
    secondary = _StubReranker({"a": 0.2, "b": 0.9, "c": 0.8})
    rr = EnsembleCrossEncoder(primary, secondary, primary_weight=0.5)
    hits = [_item("a"), _item("b"), _item("c")]
    ranked = await rr.pick_best(hits, make_query("q", gpt_target), k=3)

    assert [r.hit.record.id for r in ranked] == ["c", "a", "b"]
    # c: 0.5*0.7 + 0.5*0.8 = 0.75
    assert ranked[0].semantic_score == pytest.approx(0.75)
    # a: 0.5*0.9 + 0.5*0.2 = 0.55
    assert ranked[1].semantic_score == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_ensemble_weight_respected(gpt_target):
    """primary_weight=1.0 collapses to primary's ranking."""
    primary = _StubReranker({"a": 0.9, "b": 0.2})
    secondary = _StubReranker({"a": 0.1, "b": 0.9})
    rr = EnsembleCrossEncoder(primary, secondary, primary_weight=1.0)
    hits = [_item("a"), _item("b")]
    ranked = await rr.pick_best(hits, make_query("q", gpt_target), k=2)
    assert [r.hit.record.id for r in ranked] == ["a", "b"]


@pytest.mark.asyncio
async def test_ensemble_empty(gpt_target):
    rr = EnsembleCrossEncoder(
        _StubReranker({}), _StubReranker({}), primary_weight=0.5
    )
    assert await rr.pick_best([], make_query("q", gpt_target), k=5) == []


def test_ensemble_invalid_weight():
    with pytest.raises(ValueError):
        EnsembleCrossEncoder(
            _StubReranker({}), _StubReranker({}), primary_weight=1.5
        )
    with pytest.raises(ValueError):
        EnsembleCrossEncoder(
            _StubReranker({}), _StubReranker({}), primary_weight=-0.1
        )
