from datetime import datetime, timedelta, timezone

import pytest

from orchestration.pipeline.contracts import MemoryItem, RetrievalHit
from orchestration.reranker import ScoreReranker

from tests.conftest import make_query


def _item(id: str, *, similarity: float, days_ago: int, authority: float, pinned: bool = False):
    rec = MemoryItem(
        id=id,
        content="...",
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        source="test",
        authority_score=authority,
        pinned=pinned,
    )
    return RetrievalHit(record=rec, similarity=similarity)


@pytest.mark.asyncio
async def test_reranker_sorts_by_final_score(config, gpt_target):
    rr = ScoreReranker(config)
    hits = [
        _item("a", similarity=0.2, days_ago=200, authority=0.1),
        _item("b", similarity=0.9, days_ago=1, authority=0.9, pinned=True),
        _item("c", similarity=0.5, days_ago=30, authority=0.5),
    ]
    ranked = await rr.pick_best(hits, make_query("q", gpt_target), k=10)
    assert [r.record.id for r in ranked] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_reranker_respects_k(config, gpt_target):
    rr = ScoreReranker(config)
    hits = [_item(f"id-{i}", similarity=0.5, days_ago=i, authority=0.5) for i in range(20)]
    ranked = await rr.pick_best(hits, make_query("q", gpt_target), k=5)
    assert len(ranked) == 5


@pytest.mark.asyncio
async def test_reranker_empty_input(config, gpt_target):
    rr = ScoreReranker(config)
    assert await rr.pick_best([], make_query("q", gpt_target)) == []
