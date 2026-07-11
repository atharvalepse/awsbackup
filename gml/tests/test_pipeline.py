"""End-to-end Pipeline tests — exercising both branches of the
"found anything?" decision plus the pleasantry short-circuit.
"""
import pytest

from orchestration.classifier import KeywordClassifier
from orchestration.embedder import StubEmbedder
from orchestration.pipeline import Pipeline
from orchestration.pipeline.contracts import MemoryItem, TargetDescriptor
from orchestration.reranker import ScoreReranker
from orchestration.retriever import StubRetriever
from orchestration.sam import SAM
from orchestration.translator import Translator


def _build_pipeline(config, retriever=None):
    return Pipeline(
        classifier=KeywordClassifier(),
        embedder=StubEmbedder(dim=384),
        retriever=retriever or StubRetriever(dim=384),
        reranker=ScoreReranker(config),
        sam=SAM(),
        translator=Translator(),
        config=config,
    )


@pytest.mark.asyncio
async def test_yes_branch_end_to_end_claude(config, claude_target):
    p = _build_pipeline(config)
    q = Pipeline.build_query("how is auth_service implemented?", claude_target)
    payload = await p.run(q)
    assert payload.target.model_family.value == "claude"
    assert "<context>" in payload.formatted_context
    assert payload.metadata["items_included"] >= 1
    assert payload.metadata["target_family"] == "claude"


@pytest.mark.asyncio
async def test_no_branch_when_retriever_returns_empty(config, gpt_target):
    """High threshold so no record passes search → NO branch → reason_from_scratch."""
    retriever = StubRetriever(dim=384, match_threshold=2.0)
    p = _build_pipeline(config, retriever=retriever)
    q = Pipeline.build_query("anything goes", gpt_target)
    payload = await p.run(q)
    assert payload.metadata["items_included"] == 0
    assert payload.metadata["reason_from_scratch"] is True
    assert "reason from scratch" in payload.formatted_context.lower()


@pytest.mark.asyncio
async def test_pleasantry_short_circuit(config, gpt_target):
    p = _build_pipeline(config)
    payload = await p.run(Pipeline.build_query("hello", gpt_target))
    assert payload.metadata.get("short_circuit") == "pleasantry"
    assert payload.metadata["items_included"] == 0


@pytest.mark.asyncio
async def test_three_targets_end_to_end(config, gpt_target, gemini_target, llama_target):
    """Same query through three target families produces three differently-shaped payloads."""
    p = _build_pipeline(config)
    text = "how is auth_service implemented?"
    out_gpt = await p.run(Pipeline.build_query(text, gpt_target))
    out_gem = await p.run(Pipeline.build_query(text, gemini_target))
    out_lla = await p.run(Pipeline.build_query(text, llama_target))
    # Each adapter leaves its own signature in the rendered output.
    assert "## Context" in out_gpt.formatted_context
    assert "Retrieved Context" in out_gem.formatted_context
    assert "### Context" in out_lla.formatted_context


@pytest.mark.asyncio
async def test_pipeline_caches_per_target(config, gpt_target):
    """Per-target tokenizer + assembler + overhead are memoized across calls."""
    p = _build_pipeline(config)
    await p.run(Pipeline.build_query("question one", gpt_target))
    await p.run(Pipeline.build_query("question two", gpt_target))
    # Same (family, model_version) → one cache entry.
    assert len(p._target_cache) == 1


@pytest.mark.asyncio
async def test_pipeline_runs_to_deepseek_target(config, deepseek_target):
    """DeepSeek target family flows end-to-end and uses the DeepSeek adapter."""
    p = _build_pipeline(config)
    payload = await p.run(Pipeline.build_query("how is auth_service implemented?", deepseek_target))
    assert payload.target.model_family.value == "deepseek"
    assert "Retrieved Context" in payload.formatted_context
    assert payload.metadata["translator_adapter"] == "deepseek"


@pytest.mark.asyncio
async def test_pipeline_propagates_sam_llm_outputs(config, gpt_target):
    """When SAM has an LLM reasoner, both improved_query and reasoning_content
    flow through to the final TranslatedPayload."""
    from orchestration.sam import SAM
    from orchestration.sam._ollama_client import MockOllamaClient
    from orchestration.sam.llm_reasoner import LLMReasoner

    mock = MockOllamaClient()
    mock.queue(answer={
        "drop_ids": [],
        "improved_query": "What architectural choices shape the auth_service?",
        "reasoning": "auth_service runs FastAPI with JWT; user likely wants implementation specifics.",
    })

    p = Pipeline(
        classifier=KeywordClassifier(),
        embedder=StubEmbedder(dim=384),
        retriever=StubRetriever(dim=384),
        reranker=ScoreReranker(config),
        sam=SAM(reasoner=LLMReasoner(mock)),
        translator=Translator(),
        config=config,
    )

    payload = await p.run(Pipeline.build_query("how does auth work?", gpt_target))
    assert payload.user_query == "What architectural choices shape the auth_service?"
    assert payload.metadata["query_was_improved"] is True
    assert payload.metadata["original_user_query"] == "how does auth work?"
    assert "SAM Reasoning" in payload.formatted_context
    assert "auth_service runs FastAPI" in payload.formatted_context
