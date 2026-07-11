from datetime import datetime, timedelta, timezone

import pytest

from orchestration.assembler import BudgetAssembler
from orchestration.errors import BudgetExceededError
from orchestration.pipeline.contracts import (
    MemoryItem,
    RankedHit,
    ResolvedMemorySet,
    RetrievalHit,
)
from orchestration.tokenizers import TiktokenTokenizer

from tests.conftest import make_query


def _ranked(id: str, *, content: str, pinned: bool = False, days_ago: int = 1, score: float = 0.5,
            summary_medium: str | None = None, summary_short: str | None = None):
    rec = MemoryItem(
        id=id, content=content,
        summary_medium=summary_medium, summary_short=summary_short,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        source="test", authority_score=0.5, pinned=pinned,
    )
    return RankedHit(
        hit=RetrievalHit(record=rec, similarity=0.5),
        semantic_score=0.5, recency_score=0.5, authority_score=0.5,
        pin_boost=1.0 if pinned else 0.0, final_score=score, score_reason="test",
    )


@pytest.mark.asyncio
async def test_reason_from_scratch_produces_empty_context(config, gpt_target):
    asm = BudgetAssembler(TiktokenTokenizer("gpt-4o"), config)
    resolved = ResolvedMemorySet(reason_from_scratch=True, notes=["sentinel"])
    ctx = asm.package(resolved, make_query("hi", gpt_target), template_overhead_tokens=100)
    assert ctx.selected == []
    assert ctx.metadata["reason_from_scratch"] is True


@pytest.mark.asyncio
async def test_assembler_caps_at_final(config, gpt_target):
    asm = BudgetAssembler(TiktokenTokenizer("gpt-4o"), config)
    hits = [_ranked(f"id-{i}", content="x" * 5, score=1.0 - i * 0.01) for i in range(10)]
    resolved = ResolvedMemorySet(kept=hits)
    ctx = asm.package(resolved, make_query("q", gpt_target), template_overhead_tokens=10, final=3)
    # `final=3` plus protected (recent-N=3, none pinned, so just the top by recency overlaps)
    assert len(ctx.selected) <= 5  # at most: 3 final + edge cases from protected


@pytest.mark.asyncio
async def test_assembler_raises_budget_exceeded_on_tiny_window(config):
    asm = BudgetAssembler(TiktokenTokenizer("gpt-4o"), config)
    # Build a query whose target has a tiny window that can't fit overhead
    from orchestration.pipeline.contracts import TargetDescriptor
    tiny = TargetDescriptor.for_chatgpt(context_window=100)
    resolved = ResolvedMemorySet(kept=[])
    with pytest.raises(BudgetExceededError):
        asm.package(resolved, make_query("q", tiny), template_overhead_tokens=200)


@pytest.mark.asyncio
async def test_assembler_compresses_when_full_doesnt_fit(config, gpt_target):
    asm = BudgetAssembler(TiktokenTokenizer("gpt-4o"), config)
    hits = [
        _ranked("a", content="x " * 2000, summary_short="brief", score=0.9),
    ]
    resolved = ResolvedMemorySet(kept=hits)
    # Tight overhead pushes the full content out of budget.
    target_window = 1100  # output_reserve=275, leaves ~825, minus margin 110, ~700 budget; content is ~1000 tokens
    from orchestration.pipeline.contracts import TargetDescriptor
    target = TargetDescriptor.for_chatgpt(context_window=target_window)
    ctx = asm.package(resolved, make_query("q", target), template_overhead_tokens=10, final=1)
    assert len(ctx.selected) == 1
    # Content should have been swapped to the short summary.
    assert ctx.selected[0].record.content == "brief"
