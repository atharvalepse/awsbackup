"""Tests for orchestration/reranker/llm_reranker.py — LLM rerank with mocked client."""
import json
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import (
    MemoryItem,
    Query,
    RetrievalHit,
    TargetDescriptor,
)
from orchestration.reranker.llm_reranker import (
    LLMReranker,
    _complete_perm,
)
from orchestration.sam._ollama_client import MockOllamaClient


def _record(rid: str, content: str) -> MemoryItem:
    return MemoryItem(
        id=rid, content=content, timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.7, pinned=False,
    )


def _hit(rid: str, content: str, sim: float = 0.5) -> RetrievalHit:
    return RetrievalHit(record=_record(rid, content), similarity=sim)


def _query(text: str) -> Query:
    return Query(text=text, target=TargetDescriptor.for_claude(),
                 session_context={}, trace_id="t")


class TestCompletePerm:
    def test_complete(self):
        # All indices present already
        assert _complete_perm([2, 0, 1], n=3) == [2, 0, 1]

    def test_missing_appended(self):
        # Only [2, 0] given for n=3 → 1 should be appended
        assert _complete_perm([2, 0], n=3) == [2, 0, 1]

    def test_empty_input(self):
        assert _complete_perm([], n=3) == [0, 1, 2]


class TestParseRanking:
    def _rr(self):
        return LLMReranker(client=MockOllamaClient())

    def test_valid_ranking(self):
        rr = self._rr()
        out = rr._parse_ranking(json.dumps({"ranking": [2, 0, 1]}), n=3)
        assert out == [2, 0, 1]

    def test_partial_ranking_padded(self):
        rr = self._rr()
        # Only [1, 2] returned; 0 should be appended at the end
        out = rr._parse_ranking(json.dumps({"ranking": [1, 2]}), n=3)
        assert out == [1, 2, 0]

    def test_out_of_range_dropped(self):
        rr = self._rr()
        # 99 is out of range; should be dropped, missing 0 and 2 appended
        out = rr._parse_ranking(json.dumps({"ranking": [1, 99]}), n=3)
        assert 99 not in out
        assert set(out) == {0, 1, 2}

    def test_garbage_text_returns_identity(self):
        rr = self._rr()
        out = rr._parse_ranking("not JSON at all", n=4)
        assert out == [0, 1, 2, 3]

    def test_markdown_fence(self):
        rr = self._rr()
        out = rr._parse_ranking('```json\n{"ranking": [1, 0]}\n```', n=2)
        assert out == [1, 0]

    def test_salvage_with_integers(self):
        """If JSON parse fails, salvage integers from the text."""
        rr = self._rr()
        # Malformed JSON but ints visible
        out = rr._parse_ranking('not valid {ranking [1, 0', n=2)
        assert set(out) == {0, 1}

    def test_duplicates_in_ranking_deduped(self):
        rr = self._rr()
        out = rr._parse_ranking(json.dumps({"ranking": [1, 1, 0]}), n=2)
        assert out == [1, 0]


@pytest.mark.asyncio
class TestLLMRerankerE2E:
    async def test_basic_rerank(self):
        client = MockOllamaClient()
        client.queue(answer=json.dumps({"ranking": [2, 0, 1]}))
        rr = LLMReranker(client=client)
        hits = [
            _hit("a", "alpha"), _hit("b", "beta"), _hit("c", "gamma"),
        ]
        out = await rr.pick_best(hits, _query("anything"), k=3)
        # LLM said index 2 (=c) is best
        assert out[0].hit.record.id == "c"
        assert out[1].hit.record.id == "a"
        assert out[2].hit.record.id == "b"

    async def test_empty_hits(self):
        rr = LLMReranker(client=MockOllamaClient())
        out = await rr.pick_best([], _query("anything"), k=5)
        assert out == []

    async def test_truncates_to_max_candidates(self):
        client = MockOllamaClient()
        # 3-element ranking even though we pass 30 hits
        client.queue(answer=json.dumps({"ranking": [2, 0, 1]}))
        rr = LLMReranker(client=client, max_candidates=3)
        hits = [_hit(f"r{i}", f"content {i}") for i in range(30)]
        out = await rr.pick_best(hits, _query("anything"), k=5)
        # Only the first 3 were considered
        kept_ids = {h.hit.record.id for h in out}
        assert kept_ids <= {"r0", "r1", "r2"}

    async def test_llm_failure_returns_input_order(self):
        class _Failing:
            async def generate(self, *a, **kw):
                raise RuntimeError("ollama down")
        rr = LLMReranker(client=_Failing())
        hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
        out = await rr.pick_best(hits, _query("anything"), k=3)
        # Falls back to input order
        assert [h.hit.record.id for h in out] == ["a", "b", "c"]

    async def test_garbage_response_falls_back(self):
        client = MockOllamaClient()
        client.queue(answer="not JSON at all")
        rr = LLMReranker(client=client)
        hits = [_hit("a", "x"), _hit("b", "y")]
        out = await rr.pick_best(hits, _query("anything"), k=2)
        assert {h.hit.record.id for h in out} == {"a", "b"}

    async def test_final_score_monotonically_decreasing(self):
        client = MockOllamaClient()
        client.queue(answer=json.dumps({"ranking": [1, 0, 2]}))
        rr = LLMReranker(client=client)
        hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
        out = await rr.pick_best(hits, _query("q"), k=3)
        # final_score should be monotonically decreasing
        scores = [r.final_score for r in out]
        assert scores == sorted(scores, reverse=True)
