"""Tests for orchestration/sdp/hyde.py — HyDE rewrite/fuse with mocked client."""
import asyncio

import pytest

from orchestration.sam._ollama_client import GenerationResult, MockOllamaClient
from orchestration.sdp.hyde import hyde_fuse, hyde_rewrite


@pytest.mark.asyncio
async def test_rewrite_with_client(monkeypatch):
    """Mock client returns a hypothetical answer; rewrite returns it cleaned."""
    monkeypatch.setenv("GML_HYDE", "1")
    # Force re-evaluation of the module-level env-var (already imported)
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)

    client = MockOllamaClient()
    client.queue(answer="Caroline mentioned she's been looking into adoption agencies.")
    out = await hyde_rewrite("What did Caroline research?", client)
    assert "adoption" in out.lower()
    # No "Sure," lead-in remained
    assert not out.lower().startswith("sure")


@pytest.mark.asyncio
async def test_rewrite_strips_lead_in(monkeypatch):
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)
    client = MockOllamaClient()
    client.queue(answer="Sure! Caroline went to the support group last Sunday.")
    out = await hyde_rewrite("When did Caroline go?", client)
    assert not out.lower().startswith("sure")
    assert "caroline" in out.lower()


@pytest.mark.asyncio
async def test_disabled_returns_original(monkeypatch):
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", False)
    client = MockOllamaClient()
    # Even with a client, disabled-by-env returns the original
    out = await hyde_rewrite("anything", client)
    assert out == "anything"


@pytest.mark.asyncio
async def test_no_client_returns_original(monkeypatch):
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)
    out = await hyde_rewrite("anything", None)
    assert out == "anything"


@pytest.mark.asyncio
async def test_empty_text_returns_empty(monkeypatch):
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)
    client = MockOllamaClient()
    out = await hyde_rewrite("", client)
    assert out == ""


@pytest.mark.asyncio
async def test_llm_failure_returns_original(monkeypatch):
    """If the LLM call raises, hyde returns the original text."""
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)

    class _FailingClient:
        async def generate(self, *args, **kwargs):
            raise RuntimeError("ollama down")

    out = await hyde_rewrite("What did Caroline research?", _FailingClient())
    assert out == "What did Caroline research?"


@pytest.mark.asyncio
async def test_fuse_concats_original_plus_hypothetical(monkeypatch):
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)
    client = MockOllamaClient()
    client.queue(answer="Caroline looked into adoption agencies.")
    fused = await hyde_fuse("What did Caroline research?", client)
    # Both the question wording and the hypothetical answer's words show up
    assert "research" in fused.lower()
    assert "adoption" in fused.lower()


@pytest.mark.asyncio
async def test_fuse_returns_original_when_hyde_passthrough(monkeypatch):
    """If hyde returns original text (no rewrite), fuse also returns original."""
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", False)
    client = MockOllamaClient()
    out = await hyde_fuse("Question?", client)
    assert out == "Question?"


@pytest.mark.asyncio
async def test_very_short_rewrite_falls_back(monkeypatch):
    """Rewrite < 10 chars is treated as garbage; fall back to original."""
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)
    client = MockOllamaClient()
    client.queue(answer="ok")  # too short
    out = await hyde_rewrite("Long meaningful question?", client)
    assert out == "Long meaningful question?"


@pytest.mark.asyncio
async def test_truncation_at_max_chars(monkeypatch):
    """Long rewrites get truncated to HYDE_MAX_CHARS."""
    import orchestration.sdp.hyde as hyde_mod
    monkeypatch.setattr(hyde_mod, "HYDE_ENABLED_DEFAULT", True)
    long_answer = "Caroline " + "researched adoption agencies " * 100  # very long
    client = MockOllamaClient()
    client.queue(answer=long_answer)
    out = await hyde_rewrite("What did Caroline research?", client)
    assert len(out) <= hyde_mod.HYDE_MAX_CHARS
