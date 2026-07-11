"""Tests for the MemoryExtractor (mocked Ollama client)."""
import pytest

from orchestration.ingestion import MemoryExtractor
from orchestration.sam._ollama_client import MockOllamaClient


@pytest.mark.asyncio
async def test_extractor_parses_memories():
    mock = MockOllamaClient()
    mock.queue(answer={
        "memories": [
            {
                "content": "User prefers concise replies.",
                "entity": "user",
                "attribute": "reply_style",
                "value": "concise",
                "summary_short": "concise replies",
            },
            {
                "content": "auth_service uses FastAPI.",
                "entity": "auth_service",
                "attribute": "framework",
                "value": "FastAPI",
                "summary_short": None,
            },
        ]
    })
    extractor = MemoryExtractor(client=mock)
    items = await extractor.extract(
        # Must carry a first-person claim: the speaker-attribution guard now
        # drops batches when user_query is a pure question (the facts would be
        # coming from the assistant, not the user). This fixture exercises
        # parsing/mapping, so give it a real user claim.
        user_query="I prefer concise replies, and our auth_service uses FastAPI.",
        assistant_reply="Got it — noted your preferences.",
        session_id="s-1",
    )
    assert len(items) == 2
    assert items[0].entity == "user"
    assert items[1].value == "FastAPI"
    assert items[0].raw_metadata["session_id"] == "s-1"


@pytest.mark.asyncio
async def test_extractor_returns_empty_when_llm_fails():
    class _Broken(MockOllamaClient):
        async def generate(self, prompt, *, json_mode=False):
            raise RuntimeError("oops")

    extractor = MemoryExtractor(client=_Broken())
    items = await extractor.extract("q", "a")
    assert items == []


@pytest.mark.asyncio
async def test_assistant_fact_tagged_and_confidence_discounted():
    mock = MockOllamaClient()
    mock.queue(answer={"memories": [
        {"content": "The deploy target is staging-east-3.",
         "entity": "deploy", "attribute": "target", "value": "staging-east-3",
         "confidence": 0.9, "speaker": "assistant"},
    ]})
    extractor = MemoryExtractor(client=mock)
    # User made a first-person claim too, so the batch isn't force-assistant;
    # the per-fact speaker label drives it.
    items = await extractor.extract(
        user_query="I'm setting things up; the deploy target is what you said.",
        assistant_reply="The deploy target is staging-east-3.",
    )
    assert len(items) == 1
    assert items[0].source == "assistant"
    assert items[0].raw_metadata["speaker"] == "assistant"
    # 0.9 * 0.7 discount.
    assert abs(items[0].authority_score - 0.63) < 1e-6


@pytest.mark.asyncio
async def test_pure_question_forces_assistant_speaker():
    mock = MockOllamaClient()
    # LLM mislabels it as a user fact; the pure-question backstop overrides.
    mock.queue(answer={"memories": [
        {"content": "PostgreSQL 16 runs on port 5432.",
         "entity": "postgres", "attribute": "port", "value": "5432",
         "confidence": 0.8, "speaker": "user"},
    ]})
    extractor = MemoryExtractor(client=mock)
    items = await extractor.extract(
        # Pure question, no first-person claim → backstop forces assistant.
        user_query="what database and port does the service use?",
        assistant_reply="PostgreSQL 16 runs on port 5432.",
    )
    assert len(items) == 1
    assert items[0].source == "assistant"  # forced despite LLM's "user" label
    assert items[0].raw_metadata["speaker"] == "assistant"


@pytest.mark.asyncio
async def test_user_fact_keeps_source_and_full_confidence():
    mock = MockOllamaClient()
    mock.queue(answer={"memories": [
        {"content": "User prefers Rust.", "entity": "user",
         "attribute": "language", "value": "Rust",
         "confidence": 0.9, "speaker": "user"},
    ]})
    extractor = MemoryExtractor(client=mock)
    items = await extractor.extract(
        user_query="I prefer Rust.", assistant_reply="Noted.",
    )
    assert len(items) == 1
    assert items[0].source == "conversation"  # default, not "assistant"
    assert abs(items[0].authority_score - 0.9) < 1e-6  # no discount


@pytest.mark.asyncio
async def test_extractor_returns_empty_when_no_memories():
    mock = MockOllamaClient()
    mock.queue(answer={"memories": []})
    extractor = MemoryExtractor(client=mock)
    items = await extractor.extract("q", "a")
    assert items == []


@pytest.mark.asyncio
async def test_extractor_ignores_unparseable_output():
    mock = MockOllamaClient()
    mock.queue(answer="hello, this is not JSON at all")
    extractor = MemoryExtractor(client=mock)
    items = await extractor.extract("q", "a")
    assert items == []
