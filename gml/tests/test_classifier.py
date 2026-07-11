import os

import pytest

from orchestration.classifier import KeywordClassifier, LLMClassifier
from orchestration.pipeline.contracts import ClassificationSource

from tests.conftest import make_query


@pytest.mark.asyncio
async def test_keyword_classifier_debugging(gpt_target):
    c = KeywordClassifier()
    result = await c.classify(make_query("fix the auth bug please", gpt_target))
    assert result.intent_type == "debugging"
    assert result.source == ClassificationSource.KEYWORD_FALLBACK
    assert not result.degraded


@pytest.mark.asyncio
async def test_keyword_classifier_other(gpt_target):
    c = KeywordClassifier()
    result = await c.classify(make_query("xyzzy", gpt_target))
    assert result.intent_type == "other"


@pytest.mark.asyncio
async def test_llm_classifier_stub_mode_uses_keyword_fallback(monkeypatch, gpt_target):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    c = LLMClassifier(api_key=None)
    result = await c.classify(make_query("implement the new feature", gpt_target))
    assert result.intent_type == "coding"
    assert result.source == ClassificationSource.KEYWORD_FALLBACK
    assert not result.degraded
