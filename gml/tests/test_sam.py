from datetime import datetime, timedelta, timezone

import pytest

from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    MemoryItem,
    RankedHit,
    RetrievalHit,
)
from orchestration.pipeline.contracts import TargetDescriptor
from orchestration.sam import SAM
from orchestration.sam._ollama_client import GenerationResult, MockOllamaClient
from orchestration.sam.llm_reasoner import LLMReasoner
from orchestration.sam.resolvers import HeuristicConflictResolver, StubConflictResolver

from tests.conftest import make_query


def _default_target() -> TargetDescriptor:
    return TargetDescriptor.for_chatgpt()


def _ranked(id: str, *, value: str, days_ago: int, score: float = 0.5):
    rec = MemoryItem(
        id=id,
        content=f"{id} content",
        entity="auth_service",
        attribute="framework",
        value=value,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        source="test",
        authority_score=0.5,
    )
    return RankedHit(
        hit=RetrievalHit(record=rec, similarity=0.5),
        semantic_score=0.5, recency_score=0.5, authority_score=0.5, pin_boost=0.0,
        final_score=score, score_reason="test",
    )


@pytest.mark.asyncio
async def test_reason_from_scratch_returns_empty_with_flag(gpt_target):
    sam = SAM()
    classification = Classification(
        intent_type="question", entities=[], retrieval_hints={},
        confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
    )
    result = await sam.reason_from_scratch(make_query("q", gpt_target), classification)
    assert result.reason_from_scratch is True
    assert result.kept == []
    assert result.notes


@pytest.mark.asyncio
async def test_resolve_conflicts_drops_old_value():
    sam = SAM(conflict_resolver=HeuristicConflictResolver(), drop_threshold=0.5)
    new = _ranked("new", value="FastAPI", days_ago=1, score=0.9)
    old = _ranked("old", value="Flask", days_ago=400, score=0.5)
    result = await sam.resolve_conflicts(make_query("q", _default_target()),[new, old])
    kept_ids = {r.record.id for r in result.kept}
    assert "new" in kept_ids
    assert "old" not in kept_ids
    assert ("old", "new") in result.superseded


@pytest.mark.asyncio
async def test_resolve_conflicts_stub_keeps_everything():
    sam = SAM(conflict_resolver=StubConflictResolver())
    a = _ranked("a", value="FastAPI", days_ago=1)
    b = _ranked("b", value="Flask", days_ago=400)
    result = await sam.resolve_conflicts(make_query("q", _default_target()),[a, b])
    assert len(result.kept) == 2


@pytest.mark.asyncio
async def test_resolve_conflicts_empty():
    sam = SAM()
    result = await sam.resolve_conflicts(make_query("q", _default_target()),[])
    assert result.kept == []
    assert result.reason_from_scratch is False


# ---------------------------------------------------------------------------
# LLM-backed SAM tests (mocked OllamaClient)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_reason_from_scratch_emits_improved_query_and_reasoning(gpt_target):
    mock = MockOllamaClient()
    mock.queue(
        thinking="the user is asking about authentication; I should narrow this down",
        answer={
            "improved_query": "How does the auth_service JWT validation handle clock skew?",
            "reasoning": "JWT clock skew is a known production incident pattern; surface it.",
        },
    )
    sam = SAM(reasoner=LLMReasoner(mock))
    classification = Classification(
        intent_type="debugging", entities=["auth_service"], retrieval_hints={},
        confidence=0.8, source=ClassificationSource.LLM,
    )
    result = await sam.reason_from_scratch(make_query("fix auth bug", gpt_target), classification)
    assert result.reason_from_scratch is True
    assert result.improved_query == "How does the auth_service JWT validation handle clock skew?"
    assert "clock skew" in result.reasoning_content
    assert result.reasoner_thinking and "narrow this down" in result.reasoner_thinking


@pytest.mark.asyncio
async def test_llm_resolve_conflicts_drops_ids_returned_by_model(gpt_target):
    mock = MockOllamaClient()
    mock.queue(answer={
        "drop_ids": ["old"],
        "improved_query": "Is auth_service still on Flask or FastAPI?",
        "reasoning": "Old record contradicts newer; trust the newer.",
    })
    sam = SAM(reasoner=LLMReasoner(mock))
    new = _ranked("new", value="FastAPI", days_ago=1, score=0.9)
    old = _ranked("old", value="Flask", days_ago=400, score=0.5)
    result = await sam.resolve_conflicts(make_query("q", gpt_target), [new, old])
    kept_ids = {r.record.id for r in result.kept}
    assert kept_ids == {"new"}
    assert ("old", "new") in result.superseded
    assert result.improved_query.startswith("Is auth_service")
    assert "newer" in result.reasoning_content


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_heuristic(gpt_target):
    class BrokenClient(MockOllamaClient):
        async def generate(self, prompt: str, *, json_mode: bool = False):
            raise RuntimeError("connection refused")

    sam = SAM(reasoner=LLMReasoner(BrokenClient()))
    new = _ranked("new", value="FastAPI", days_ago=1, score=0.9)
    old = _ranked("old", value="Flask", days_ago=400, score=0.5)
    result = await sam.resolve_conflicts(make_query("q", gpt_target), [new, old])
    # Heuristic fallback still drops the older record
    kept_ids = {r.record.id for r in result.kept}
    assert "new" in kept_ids
    assert "old" not in kept_ids
    # LLM-only fields stay empty in fallback
    assert result.improved_query is None
    assert result.reasoning_content is None


@pytest.mark.asyncio
async def test_llm_drop_ids_filters_unknown(gpt_target):
    """Defense: model hallucinates an id not in the input — SAM must ignore it."""
    mock = MockOllamaClient()
    mock.queue(answer={"drop_ids": ["nonexistent"], "improved_query": "q'", "reasoning": "r"})
    sam = SAM(reasoner=LLMReasoner(mock))
    new = _ranked("new", value="FastAPI", days_ago=1, score=0.9)
    result = await sam.resolve_conflicts(make_query("q", gpt_target), [new])
    assert [r.record.id for r in result.kept] == ["new"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_with_real_ollama_deepseek_r1():
    """Real Ollama call. Skipped automatically when the daemon isn't reachable."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            r.raise_for_status()
    except Exception:
        pytest.skip("Ollama daemon not reachable at localhost:11434")

    sam = SAM.with_ollama(timeout_seconds=60.0)
    classification = Classification(
        intent_type="debugging", entities=["auth_service"], retrieval_hints={},
        confidence=0.8, source=ClassificationSource.LLM,
    )
    result = await sam.reason_from_scratch(
        make_query("fix the auth bug", _default_target()), classification
    )
    assert result.reason_from_scratch is True
    # Real DeepSeek R1 produces non-empty reasoning + improved query
    assert result.improved_query, "DeepSeek should return a non-empty improved_query"
    assert result.reasoning_content, "DeepSeek should return non-empty reasoning"
