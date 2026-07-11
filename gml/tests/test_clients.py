"""Tests for the target-AI client layer.

Unit tests mock each SDK / HTTP client. Integration tests (skippable)
hit live services.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.clients import (
    AnthropicClient,
    GeminiClient,
    OllamaClient,
    OpenAIClient,
    StubClient,
    build_default_client_for_target,
)
from orchestration.clients.anthropic_client import AnthropicClientError
from orchestration.clients.gemini_client import GeminiClientError
from orchestration.clients.ollama_client import OllamaClientError
from orchestration.clients.openai_client import OpenAIClientError
from orchestration.pipeline.contracts import ModelFamily, TranslatedPayload


def _payload(target):
    return TranslatedPayload(
        formatted_context="context here",
        user_query="how does it work?",
        target=target,
        trace_id="t-1",
        config_hash="abc",
        metadata={},
    )


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_client_records_and_echoes(claude_target):
    c = StubClient(response_text="canned reply")
    resp = await c.send(_payload(claude_target))
    assert resp.text == "canned reply"
    assert len(c.received) == 1
    assert resp.target.model_family == ModelFamily.CLAUDE


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_client_raises_without_key(monkeypatch, claude_target):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = AnthropicClient(api_key=None)
    with pytest.raises(AnthropicClientError, match="ANTHROPIC_API_KEY"):
        await c.send(_payload(claude_target))


@pytest.mark.asyncio
async def test_anthropic_client_parses_response(claude_target):
    block = MagicMock()
    block.type = "text"
    block.text = "the answer"
    fake_resp = MagicMock()
    fake_resp.content = [block]
    fake_resp.stop_reason = "end_turn"
    fake_resp.usage = MagicMock(input_tokens=42, output_tokens=7)

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=fake_resp)

    with patch("anthropic.AsyncAnthropic", return_value=fake_client):
        c = AnthropicClient(api_key="x")
        resp = await c.send(_payload(claude_target))
    assert resp.text == "the answer"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 7
    assert resp.raw_metadata["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_client_parses_response(gpt_target):
    choice = MagicMock()
    choice.message.content = "gpt says hi"
    choice.finish_reason = "stop"
    fake_resp = MagicMock()
    fake_resp.choices = [choice]
    fake_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=4)

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_resp)

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        c = OpenAIClient(api_key="x")
        resp = await c.send(_payload(gpt_target))
    assert resp.text == "gpt says hi"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 4


@pytest.mark.asyncio
async def test_openai_client_raises_without_key(monkeypatch, gpt_target):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = OpenAIClient(api_key=None)
    with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
        await c.send(_payload(gpt_target))


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_client_parses_response(gemini_target):
    fake_resp = MagicMock()
    fake_resp.text = "gemini says hello"
    fake_resp.usage_metadata = MagicMock(prompt_token_count=8, candidates_token_count=3)

    fake_aio = MagicMock()
    fake_aio.models.generate_content = AsyncMock(return_value=fake_resp)
    fake_client = MagicMock()
    fake_client.aio = fake_aio

    with patch("google.genai.Client", return_value=fake_client):
        c = GeminiClient(api_key="x")
        resp = await c.send(_payload(gemini_target))
    assert resp.text == "gemini says hello"
    assert resp.input_tokens == 8


@pytest.mark.asyncio
async def test_gemini_client_raises_without_key(monkeypatch, gemini_target):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    c = GeminiClient(api_key=None)
    with pytest.raises(GeminiClientError, match="GEMINI_API_KEY"):
        await c.send(_payload(gemini_target))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_client_strips_think_block(deepseek_target):
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={
        "response": "<think>internal reasoning</think>actual answer",
        "prompt_eval_count": 12, "eval_count": 6, "done": True,
    })

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.post = AsyncMock(return_value=fake_resp)

    with patch("httpx.AsyncClient", return_value=fake_client):
        c = OllamaClient()
        resp = await c.send(_payload(deepseek_target))
    assert resp.text == "actual answer"
    assert "internal reasoning" in resp.raw_metadata["raw_response_with_thinking"]
    assert resp.input_tokens == 12


@pytest.mark.asyncio
async def test_ollama_client_errors_when_daemon_unreachable(deepseek_target):
    c = OllamaClient(base_url="http://127.0.0.1:1", timeout_seconds=0.5)
    with pytest.raises(OllamaClientError):
        await c.send(_payload(deepseek_target))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_dispatches_by_family(gpt_target, claude_target, gemini_target, llama_target, deepseek_target):
    assert isinstance(build_default_client_for_target(gpt_target), OpenAIClient)
    assert isinstance(build_default_client_for_target(claude_target), AnthropicClient)
    assert isinstance(build_default_client_for_target(gemini_target), GeminiClient)
    assert isinstance(build_default_client_for_target(llama_target), OllamaClient)
    assert isinstance(build_default_client_for_target(deepseek_target), OllamaClient)
