"""Tests for orchestration/sdp/query_router.py — classify_query heuristic."""
import pytest

from orchestration.sdp.query_router import QueryHints, classify_query


class TestClassifyQuery:
    def test_empty(self):
        hints = classify_query("")
        assert hints.category == 1
        assert hints.is_temporal is False
        assert hints.is_multi_hop is False
        assert hints.top_k_multiplier == 1.0

    def test_temporal_when_question(self):
        hints = classify_query("When did Caroline go to the support group?")
        assert hints.is_temporal is True
        assert hints.category == 3
        assert hints.top_k_multiplier == 1.5
        assert "temporal_question" in hints.notes

    def test_temporal_before_after(self):
        for q in [
            "What happened before the migration?",
            "After they moved, what changed?",
            "What was the last thing they discussed?",
        ]:
            hints = classify_query(q)
            assert hints.is_temporal is True, f"Failed on: {q!r}"

    def test_multi_hop_both_signal(self):
        hints = classify_query("Did they both go to the same place?")
        assert hints.is_multi_hop is True
        assert hints.category == 2

    def test_count_question(self):
        hints = classify_query("How many sessions did Caroline attend?")
        assert hints.is_count is True
        assert hints.top_k_multiplier >= 2.0

    def test_negation(self):
        hints = classify_query("They didn't use Stripe anymore, right?")
        assert hints.is_negation is True

    def test_simple_factual_lookup(self):
        hints = classify_query("What is the auth service language?")
        assert hints.is_temporal is False
        assert hints.is_multi_hop is False
        assert hints.is_count is False
        assert hints.top_k_multiplier == 1.0

    def test_combined_temporal_count(self):
        """A question can hit multiple signals."""
        hints = classify_query("How many times before the migration did they switch?")
        assert hints.is_temporal is True
        assert hints.is_count is True
        # Multipliers shouldn't double; we take the max
        assert hints.top_k_multiplier >= 2.0

    def test_negation_doesnt_break_other_signals(self):
        hints = classify_query("When did they not use Stripe?")
        assert hints.is_temporal is True
        assert hints.is_negation is True

    def test_returns_hints_dataclass(self):
        hints = classify_query("test")
        assert isinstance(hints, QueryHints)
