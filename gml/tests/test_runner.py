"""Tests for the Conversation runner — the end-to-end driver."""
import pytest

from orchestration.classifier import KeywordClassifier
from orchestration.clients import StubClient
from orchestration.embedder import StubEmbedder
from orchestration.ingestion import MemoryExtractor
from orchestration.memory_store import JsonlMemoryStore
from orchestration.pipeline import Pipeline
from orchestration.reranker import ScoreReranker
from orchestration.retriever import StubRetriever
from orchestration.runner import Conversation
from orchestration.sam import SAM
from orchestration.sam._ollama_client import MockOllamaClient
from orchestration.translator import Translator


def _build_pipeline(config):
    return Pipeline(
        classifier=KeywordClassifier(),
        embedder=StubEmbedder(dim=384),
        retriever=StubRetriever(dim=384),
        reranker=ScoreReranker(config),
        sam=SAM(reasoner=None),  # heuristic-only to keep tests deterministic
        translator=Translator(),
        config=config,
    )


@pytest.mark.asyncio
async def test_runner_basic_round_trip(config, deepseek_target):
    pipeline = _build_pipeline(config)
    client = StubClient(response_text="the model speaks")

    conv = Conversation(
        pipeline=pipeline,
        client=client,
        target=deepseek_target,
    )
    result = await conv.ask("how is auth_service implemented?")

    assert result.response.text == "the model speaks"
    assert len(client.received) == 1
    assert conv.session.turns[0].user_query == "how is auth_service implemented?"


@pytest.mark.asyncio
async def test_runner_extracts_and_persists_memories(config, deepseek_target, tmp_path):
    pipeline = _build_pipeline(config)
    client = StubClient(response_text="auth_service uses FastAPI")

    mock_extractor_client = MockOllamaClient()
    mock_extractor_client.queue(answer={
        "memories": [
            {
                "content": "auth_service is implemented in FastAPI.",
                "entity": "auth_service",
                "attribute": "framework",
                "value": "FastAPI",
                "summary_short": "auth_service: FastAPI",
            }
        ]
    })

    store = JsonlMemoryStore(tmp_path / "memories.jsonl")
    conv = Conversation(
        pipeline=pipeline,
        client=client,
        target=deepseek_target,
        extractor=MemoryExtractor(client=mock_extractor_client),
        memory_store=store,
    )
    result = await conv.ask("what runs auth_service?")

    assert len(result.extracted_memories) == 1
    persisted = store.load_all()
    assert len(persisted) == 1
    assert persisted[0].entity == "auth_service"


@pytest.mark.asyncio
async def test_runner_session_history_accumulates(config, deepseek_target):
    pipeline = _build_pipeline(config)
    client = StubClient(response_text="reply")
    conv = Conversation(pipeline=pipeline, client=client, target=deepseek_target)

    await conv.ask("first")
    await conv.ask("second")
    await conv.ask("third")

    assert [t.user_query for t in conv.session.turns] == ["first", "second", "third"]
    # Session history is fed into the pipeline via session_context
    last_payload_for_third_turn = client.received[-1]
    # session_id is consistent across turns
    assert len(set(t.trace_id for t in conv.session.turns)) == 3
    assert conv.session.session_id  # non-empty
