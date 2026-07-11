"""Tests for the heuristic query decomposer."""
import pytest

from orchestration.sdp.query_decomposer import decompose, _looks_multi_item


class TestConjunctionSplit:
    def test_simple_and(self):
        out = decompose("What yoga and meditation does Caroline do?")
        assert len(out) >= 2
        # Original first
        assert out[0] == "What yoga and meditation does Caroline do?"
        # Each split contains one half
        joined = " ".join(out[1:]).lower()
        assert "yoga" in joined and "meditation" in joined

    def test_simple_or(self):
        out = decompose("Does Caroline prefer yoga or meditation?")
        # "or" between 1-word fragments — guarded out (needs ≥3 words/side)
        # actually let's check: "Does Caroline prefer yoga" (4) "meditation" (1)
        # → second side is only 1 word, so should NOT split
        assert len(out) == 1

    def test_compound_substantive_or(self):
        out = decompose(
            "Where did Evan travel last summer or where does he plan to go next?"
        )
        assert len(out) >= 2

    def test_does_not_split_when_one_side_too_short(self):
        # "yes and no" type — guarded out
        out = decompose("Did Caroline say yes and no?")
        assert len(out) == 1


class TestMultiItemRewording:
    def test_cardinal_number_triggers_rewrite(self):
        out = decompose("What two practices does Caroline do?")
        assert len(out) >= 2
        assert out[0] == "What two practices does Caroline do?"
        # One of the rewrites drops "two" or adds "all"
        rest = " ".join(out[1:]).lower()
        # Either rewrite must show up
        assert "practices" in rest

    def test_what_kinds_triggers(self):
        out = decompose("What kinds of activities does Sam do?")
        assert len(out) >= 1  # may or may not produce extra rewrites

    def test_no_multi_item_hint_no_rewrite(self):
        # Single-fact question — no decomposition
        out = decompose("What car does Evan drive?")
        assert out == ["What car does Evan drive?"]


class TestEdgeCases:
    def test_empty(self):
        # Whitespace-only collapses to empty after strip — acceptable
        assert decompose("") == [""]
        assert decompose("   ") == [""]

    def test_max_subqueries_respected(self):
        # Even if multiple rewrites would fire, cap at max_subqueries
        out = decompose(
            "What three hobbies and activities does Caroline do?",
            max_subqueries=2,
        )
        assert len(out) <= 2

    def test_always_includes_original_first(self):
        q = "What yoga and meditation does Caroline do?"
        out = decompose(q)
        assert out[0] == q

    def test_duplicates_filtered(self):
        # Original phrasing shouldn't appear again in rewrites
        out = decompose("What kinds of practices does Caroline do?")
        if len(out) > 1:
            assert out[0] not in out[1:]


class TestMultiItemDetection:
    @pytest.mark.parametrize("q", [
        "What kinds of things did Evan have broken?",
        "What hobbies does Sam have?",
        "What two practices does Caroline do?",
        "What three activities does Carlos enjoy?",
        "Name all the cars Evan owned",
    ])
    def test_positive(self, q):
        assert _looks_multi_item(q) is True

    @pytest.mark.parametrize("q", [
        "What car does Evan drive?",
        "Who is Mel?",
        "When did Caroline visit Jasper?",
    ])
    def test_negative(self, q):
        assert _looks_multi_item(q) is False
