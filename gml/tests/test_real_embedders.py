"""Tests for the real Embedder backends — mocked unit + skipped live integration."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.embedder import GeminiEmbedder, OllamaEmbedder
from orchestration.errors import EmbedderError
from orchestration.pipeline.contracts import Classification, ClassificationSource

from tests.conftest import make_query


def _classification():
    return Classification(
        intent_type="other", entities=[], retrieval_hints={},
        confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
    )


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_embedder_raises_when_no_api_key(monkeypatch, gpt_target):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    e = GeminiEmbedder(api_key=None)
    with pytest.raises(EmbedderError, match="GEMINI_API_KEY"):
        await e.embed(make_query("hello", gpt_target), _classification())


@pytest.mark.asyncio
async def test_gemini_embedder_extracts_vector_from_sdk_response(gpt_target):
    """Mocked SDK — verify we parse response.embeddings[0].values correctly."""
    fake_embedding = MagicMock()
    fake_embedding.values = [0.1, 0.2, 0.3, 0.4]
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]

    fake_aio = MagicMock()
    fake_aio.models.embed_content = AsyncMock(return_value=fake_response)
    fake_client = MagicMock()
    fake_client.aio = fake_aio

    with patch("google.genai.Client", return_value=fake_client):
        e = GeminiEmbedder(api_key="fake-key")
        out = await e.embed(make_query("hello", gpt_target), _classification())

    assert out.vector == [0.1, 0.2, 0.3, 0.4]
    assert out.embedder_version.startswith("gemini:")


@pytest.mark.asyncio
async def test_gemini_embedder_wraps_sdk_failure(gpt_target):
    fake_aio = MagicMock()
    fake_aio.models.embed_content = AsyncMock(side_effect=RuntimeError("network down"))
    fake_client = MagicMock()
    fake_client.aio = fake_aio

    with patch("google.genai.Client", return_value=fake_client):
        e = GeminiEmbedder(api_key="fake-key")
        with pytest.raises(EmbedderError, match="network down"):
            await e.embed(make_query("hello", gpt_target), _classification())


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_embedder_parses_embedding_field(gpt_target):
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={"embedding": [0.5, 0.6, 0.7]})

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.post = AsyncMock(return_value=fake_resp)

    with patch("httpx.AsyncClient", return_value=fake_client):
        e = OllamaEmbedder()
        out = await e.embed(make_query("hello", gpt_target), _classification())
    assert out.vector == [0.5, 0.6, 0.7]
    assert out.embedder_version == "ollama:nomic-embed-text"


@pytest.mark.asyncio
async def test_ollama_embedder_errors_when_daemon_unreachable(gpt_target):
    """Real call against an invalid base_url — guaranteed to fail."""
    e = OllamaEmbedder(base_url="http://127.0.0.1:1", timeout_seconds=0.5)
    with pytest.raises(EmbedderError):
        await e.embed(make_query("hello", gpt_target), _classification())


# ---------------------------------------------------------------------------
# Integration: hits the real local Ollama daemon. Requires nomic-embed-text
# pulled. Skipped automatically when unreachable.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_embedder_real_call(gpt_target):
    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = {m.get("name", "").split(":")[0] for m in tags}
            if "nomic-embed-text" not in names:
                pytest.skip("nomic-embed-text not pulled; `ollama pull nomic-embed-text` first")
    except Exception:
        pytest.skip("Ollama daemon not reachable at localhost:11434")

    e = OllamaEmbedder()
    out = await e.embed(make_query("authentication service", gpt_target), _classification())
    assert len(out.vector) >= 256, "real embedding should have many dims"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_embedder_real_call(gpt_target):
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    e = GeminiEmbedder()
    out = await e.embed(make_query("authentication service", gpt_target), _classification())
    # gemini-embedding-001 returns 3072-dim by default
    assert len(out.vector) >= 768
