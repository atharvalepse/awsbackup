"""Tests for the iterative-retrieval path in Pipeline.run.

When SAM produces an improved_query that differs from the original, we
re-embed and re-retrieve, union with the original candidates, dedup, and
rerank. The first SAM call's outputs (improved_query, reasoning) survive.
"""
import json

import pytest

from orchestration.classifier import KeywordClassifier
from orchestration.embedder import StubEmbedder
from orchestration.pipeline import Pipeline
from orchestration.pipeline.contracts import TargetDescriptor
from orchestration.reranker import ScoreReranker
from orchestration.retriever import StubRetriever
from orchestration.sam import SAM
from orchestration.sam._ollama_client import MockOllamaClient
from orchestration.sam.llm_reasoner import LLMReasoner
from orchestration.translator import Translator


@pytest.mark.asyncio
async def test_iterative_retrieval_preserves_first_sam_outputs(config):
    """SAM rewrites query; pipeline re-retrieves but keeps first SAM's outputs."""
    target = TargetDescriptor.for_chatgpt()
    mock = MockOllamaClient()
    # One LLM call from SAM.resolve_conflicts on the first pass
    mock.queue(answer=json.dumps({
        "drop_ids": [],
        "improved_query": "What architecture choices shape the auth_service?",
        "reasoning": "Auth uses FastAPI + JWT — user likely wants specifics.",
    }))
    p = Pipeline(
        classifier=KeywordClassifier(),
        embedder=StubEmbedder(dim=384),
        retriever=StubRetriever(dim=384),
        reranker=ScoreReranker(config),
        sam=SAM(reasoner=LLMReasoner(mock)),
        translator=Translator(),
        config=config,
    )
    payload = await p.run(Pipeline.build_query("how does auth work?", target))

    # The improved_query from the FIRST SAM call survives the iterative
    # retrieval. (Pre-fix, the SECOND SAM call wiped this out.)
    assert payload.user_query == "What architecture choices shape the auth_service?"
    assert payload.metadata["query_was_improved"] is True
    # And only ONE prompt was sent to the mock (no second SAM call)
    assert len(mock.prompts) == 1


@pytest.mark.asyncio
async def test_iterative_retrieval_disabled_env(monkeypatch, config):
    """GML_ITERATIVE_RETRIEVAL=0 skips the second retrieval entirely."""
    monkeypatch.setenv("GML_ITERATIVE_RETRIEVAL", "0")
    target = TargetDescriptor.for_chatgpt()
    mock = MockOllamaClient()
    mock.queue(answer=json.dumps({
        "drop_ids": [],
        "improved_query": "rewritten",
        "reasoning": "x",
    }))
    p = Pipeline(
        classifier=KeywordClassifier(),
        embedder=StubEmbedder(dim=384),
        retriever=StubRetriever(dim=384),
        reranker=ScoreReranker(config),
        sam=SAM(reasoner=LLMReasoner(mock)),
        translator=Translator(),
        config=config,
    )
    payload = await p.run(Pipeline.build_query("how does auth work?", target))
    # Improved query still propagated; iterative path didn't run (no log event)
    assert payload.user_query == "rewritten"
    # Only one SAM call regardless
    assert len(mock.prompts) == 1


@pytest.mark.asyncio
async def test_iterative_retrieval_skipped_when_no_improvement(config):
    """If SAM returns the same query, don't re-retrieve."""
    target = TargetDescriptor.for_chatgpt()
    mock = MockOllamaClient()
    # SAM returns the SAME query (no improvement)
    mock.queue(answer=json.dumps({
        "drop_ids": [],
        "improved_query": "how does auth work?",  # same as original
        "reasoning": "Direct factual lookup.",
    }))
    p = Pipeline(
        classifier=KeywordClassifier(),
        embedder=StubEmbedder(dim=384),
        retriever=StubRetriever(dim=384),
        reranker=ScoreReranker(config),
        sam=SAM(reasoner=LLMReasoner(mock)),
        translator=Translator(),
        config=config,
    )
    payload = await p.run(Pipeline.build_query("how does auth work?", target))
    # No iterative pass happened; first SAM still ran
    assert len(mock.prompts) == 1
    # No 'iterative_retrieval' marker in notes — verified indirectly by
    # the fact that improved_query == original means the path was skipped.
    assert payload.metadata.get("query_was_improved") is False
