"""Tests for orchestration/sam/turn_compressor.py — SAMTurnCompressor."""
import json
from datetime import datetime, timezone

import pytest

from orchestration.sam._ollama_client import MockOllamaClient
from orchestration.sam.aal_record import AALRecord
from orchestration.sam.turn_compressor import (
    SAMTurnCompressor,
    _normalize_tuple_dict,
    _parse_aal_response,
)


def _valid_response(
    tuples=None, summary="x", entities=None, importance=0.7, confidence=0.85,
) -> str:
    return json.dumps({
        "tuples": tuples or [],
        "chunk_summary": summary,
        "entities": entities or [],
        "importance": importance,
        "confidence": confidence,
    })


class TestParseAALResponse:
    def test_basic(self):
        out = _parse_aal_response(_valid_response(summary="hi"))
        assert out["chunk_summary"] == "hi"

    def test_markdown_fenced(self):
        out = _parse_aal_response("```json\n" + _valid_response(summary="hi") + "\n```")
        assert out["chunk_summary"] == "hi"

    def test_garbage(self):
        assert _parse_aal_response("not JSON") is None

    def test_empty(self):
        assert _parse_aal_response("") is None

    def test_outer_braces_only(self):
        # Trailing prose around the JSON
        out = _parse_aal_response(
            "Here is the result:\n" + _valid_response() + "\nThanks!"
        )
        assert out is not None


class TestNormalizeTupleDict:
    def test_valid(self):
        t = _normalize_tuple_dict({"subject": "X", "verb": "GO", "object": "Y"})
        assert t is not None
        assert t.verb == "go"  # lowercased

    def test_missing_field(self):
        assert _normalize_tuple_dict({"subject": "X", "verb": "go"}) is None
        assert _normalize_tuple_dict({"subject": "", "verb": "go", "object": "Y"}) is None

    def test_placeholder_subject_filtered(self):
        assert _normalize_tuple_dict({"subject": "Speaker A", "verb": "say", "object": "hi"}) is None
        assert _normalize_tuple_dict({"subject": "someone", "verb": "do", "object": "x"}) is None

    def test_negation_preserved(self):
        t = _normalize_tuple_dict({"subject": "X", "verb": "use", "object": "S", "negated": True})
        assert t.negated is True


@pytest.mark.asyncio
class TestSAMTurnCompressorE2E:
    async def test_full_compression(self):
        client = MockOllamaClient()
        client.queue(answer=_valid_response(
            tuples=[
                {"subject": "Caroline", "verb": "research",
                 "object": "agencies", "time": "weekend", "negated": False},
            ],
            summary="Caroline researched agencies",
            entities=["caroline", "agencies"],
            importance=0.85, confidence=0.9,
        ))
        compressor = SAMTurnCompressor(client=client)
        rec = await compressor.compress(
            user_text="I've been looking into adoption agencies",
            assistant_text="Caroline mentioned researching agencies",
        )
        assert len(rec.tuples) == 1
        assert rec.tuples[0].subject == "Caroline"
        assert rec.chunk_summary == "Caroline researched agencies"
        assert "caroline" in rec.entities
        assert rec.importance == 0.85
        assert rec.confidence == 0.9
        assert rec.is_empty is False

    async def test_empty_turn_returns_empty_record(self):
        client = MockOllamaClient()
        compressor = SAMTurnCompressor(client=client)
        rec = await compressor.compress(user_text="", assistant_text="")
        assert rec.is_empty is True

    async def test_pleasantry_response(self):
        """Model returns explicit empty record for pleasantries."""
        client = MockOllamaClient()
        client.queue(answer=json.dumps({
            "tuples": [], "chunk_summary": "", "entities": [],
            "importance": 0.1, "confidence": 0.5,
        }))
        compressor = SAMTurnCompressor(client=client)
        rec = await compressor.compress(user_text="hi", assistant_text="hello")
        assert rec.is_empty is True

    async def test_llm_failure_returns_safe_record(self):
        class _Failing:
            async def generate(self, *a, **kw):
                raise RuntimeError("down")
        compressor = SAMTurnCompressor(client=_Failing())
        rec = await compressor.compress(user_text="u", assistant_text="a")
        assert rec.is_empty is True
        # User/assistant texts are preserved for replay/audit even on LLM fail
        assert rec.chunk_user == "u"
        assert rec.chunk_assistant == "a"

    async def test_garbage_response_returns_safe_record(self):
        client = MockOllamaClient()
        client.queue(answer="not JSON at all")
        compressor = SAMTurnCompressor(client=client)
        rec = await compressor.compress(user_text="u", assistant_text="a")
        assert rec.is_empty is True

    async def test_importance_clamped_to_unit_range(self):
        client = MockOllamaClient()
        client.queue(answer=_valid_response(
            summary="x", importance=5.0, confidence=-0.5,  # out of [0,1]
        ))
        compressor = SAMTurnCompressor(client=client)
        rec = await compressor.compress(user_text="u", assistant_text="a")
        assert 0.0 <= rec.importance <= 1.0
        assert 0.0 <= rec.confidence <= 1.0
