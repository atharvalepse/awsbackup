import math

import pytest

from orchestration.embedder import StubEmbedder
from orchestration.pipeline.contracts import Classification, ClassificationSource

from tests.conftest import make_query


@pytest.fixture
def classification():
    return Classification(
        intent_type="other",
        entities=[],
        retrieval_hints={},
        confidence=0.5,
        source=ClassificationSource.KEYWORD_FALLBACK,
    )


@pytest.mark.asyncio
async def test_stub_embedder_deterministic(gpt_target, classification):
    e = StubEmbedder(dim=128)
    q = make_query("hello world", gpt_target)
    v1 = await e.embed(q, classification)
    v2 = await e.embed(q, classification)
    assert v1.vector == v2.vector
    assert len(v1.vector) == 128


@pytest.mark.asyncio
async def test_stub_embedder_l2_normalized(gpt_target, classification):
    e = StubEmbedder(dim=128)
    q = make_query("anything goes", gpt_target)
    out = await e.embed(q, classification)
    norm = math.sqrt(sum(x * x for x in out.vector))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_stub_embedder_different_text_different_vector(gpt_target, classification):
    e = StubEmbedder(dim=64)
    v1 = await e.embed(make_query("alpha", gpt_target), classification)
    v2 = await e.embed(make_query("beta", gpt_target), classification)
    assert v1.vector != v2.vector
