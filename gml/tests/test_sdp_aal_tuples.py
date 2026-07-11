"""Tests for orchestration/sdp/aal_tuples.py — AAL tuple extractor."""
import json

import pytest

from orchestration.sam._ollama_client import MockOllamaClient
from orchestration.sdp.aal_tuples import (
    AALTupleExtractor,
    _parse_tuples,
    tuple_to_content,
)


class TestParseTuples:
    def test_empty(self):
        assert _parse_tuples("") == []
        assert _parse_tuples("   ") == []

    def test_basic_json(self):
        payload = json.dumps({
            "tuples": [
                {"subject": "Caroline", "verb": "research", "object": "adoption agencies",
                 "time": "weekend", "negated": False},
            ]
        })
        out = _parse_tuples(payload)
        assert len(out) == 1
        assert out[0]["subject"] == "Caroline"
        assert out[0]["verb"] == "research"
        assert out[0]["object"] == "adoption agencies"

    def test_markdown_fenced_json(self):
        payload = '```json\n{"tuples": [{"subject": "X", "verb": "go", "object": "home"}]}\n```'
        out = _parse_tuples(payload)
        assert len(out) == 1
        assert out[0]["subject"] == "X"

    def test_garbage_text(self):
        assert _parse_tuples("not JSON at all") == []

    def test_missing_fields_skipped(self):
        payload = json.dumps({"tuples": [
            {"subject": "X", "verb": "go"},  # no object
            {"subject": "Y", "verb": "see", "object": "Z"},  # valid
        ]})
        out = _parse_tuples(payload)
        assert len(out) == 1
        assert out[0]["subject"] == "Y"

    def test_placeholder_subjects_filtered(self):
        payload = json.dumps({"tuples": [
            {"subject": "Speaker A", "verb": "say", "object": "hi"},
            {"subject": "someone", "verb": "do", "object": "thing"},
            {"subject": "Caroline", "verb": "go", "object": "home"},
        ]})
        out = _parse_tuples(payload)
        assert len(out) == 1
        assert out[0]["subject"] == "Caroline"

    def test_verb_normalized_to_lowercase(self):
        payload = json.dumps({"tuples": [
            {"subject": "Caroline", "verb": "RESEARCH", "object": "agencies"},
        ]})
        out = _parse_tuples(payload)
        assert out[0]["verb"] == "research"

    def test_default_negated_false(self):
        payload = json.dumps({"tuples": [
            {"subject": "X", "verb": "go", "object": "home"},
        ]})
        out = _parse_tuples(payload)
        assert out[0]["negated"] is False

    def test_negation_preserved(self):
        payload = json.dumps({"tuples": [
            {"subject": "X", "verb": "use", "object": "Stripe", "negated": True},
        ]})
        out = _parse_tuples(payload)
        assert out[0]["negated"] is True

    def test_non_list_tuples_returns_empty(self):
        payload = json.dumps({"tuples": "not a list"})
        assert _parse_tuples(payload) == []

    def test_extra_braces_tolerated(self):
        # Model sometimes emits trailing prose
        payload = 'Here is the JSON:\n{"tuples": [{"subject": "X", "verb": "go", "object": "home"}]}\nThat is all.'
        out = _parse_tuples(payload)
        assert len(out) == 1


class TestTupleToContent:
    def test_basic(self):
        t = {"subject": "Caroline", "verb": "research", "object": "agencies",
             "time": None, "negated": False}
        assert tuple_to_content(t) == "Caroline research agencies"

    def test_with_time(self):
        t = {"subject": "X", "verb": "go", "object": "Y", "time": "Sunday", "negated": False}
        assert tuple_to_content(t) == "X go Y (Sunday)"

    def test_negated(self):
        t = {"subject": "X", "verb": "use", "object": "Stripe", "time": None, "negated": True}
        out = tuple_to_content(t)
        assert "did not" in out
        assert "Stripe" in out


@pytest.mark.asyncio
class TestAALTupleExtractorE2E:
    async def test_extract_returns_memory_items(self):
        client = MockOllamaClient()
        client.queue(answer=json.dumps({
            "tuples": [
                {"subject": "Caroline", "verb": "research", "object": "adoption agencies"},
                {"subject": "Melanie", "verb": "leave", "object": "the team"},
            ]
        }))
        extractor = AALTupleExtractor(client=client)
        items = await extractor.extract_from_turn(
            user_text="anything", assistant_text="anything",
        )
        assert len(items) == 2
        # MemoryItem fields populated correctly
        assert items[0].entity == "Caroline"
        assert items[0].attribute == "research"
        assert items[0].value == "adoption agencies"
        assert items[0].source == "aal-tuple"

    async def test_extract_handles_llm_failure(self):
        class _Failing:
            async def generate(self, *a, **kw):
                raise RuntimeError("down")
        extractor = AALTupleExtractor(client=_Failing())
        items = await extractor.extract_from_turn("u", "a")
        assert items == []

    async def test_extract_handles_garbage_response(self):
        client = MockOllamaClient()
        client.queue(answer="this is not JSON")
        extractor = AALTupleExtractor(client=client)
        items = await extractor.extract_from_turn("u", "a")
        assert items == []

    async def test_empty_tuples_returns_empty(self):
        client = MockOllamaClient()
        client.queue(answer=json.dumps({"tuples": []}))
        extractor = AALTupleExtractor(client=client)
        items = await extractor.extract_from_turn("u", "a")
        assert items == []
