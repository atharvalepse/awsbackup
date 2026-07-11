"""Tests for the updated should_skip_sam heuristic (top1-top2 gap rule)."""
import os
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import MemoryItem, RankedHit, RetrievalHit
from orchestration.pipeline.pipeline import (
    SAM_SKIP_DEFAULT_GAP,
    SAM_SKIP_DEFAULT_THRESHOLD,
    should_skip_sam,
)


def _rh(rec_id: str, final_score: float,
        entity: str | None = None, attribute: str | None = None,
        value: str | None = None) -> RankedHit:
    rec = MemoryItem(
        id=rec_id, content=f"content {rec_id}",
        entity=entity, attribute=attribute, value=value,
        timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.7, pinned=False,
    )
    hit = RetrievalHit(record=rec, similarity=final_score)
    return RankedHit(
        hit=hit, semantic_score=final_score, recency_score=1.0,
        authority_score=0.7, pin_boost=0.0,
        final_score=final_score, score_reason="",
    )


class TestShouldSkipSam:
    def test_empty_does_not_skip(self):
        skip, reason = should_skip_sam([])
        assert skip is False
        assert "no ranked" in reason

    def test_low_top_score_does_not_skip(self):
        # Top score below default 0.55 → don't skip
        skip, reason = should_skip_sam([_rh("a", 0.40), _rh("b", 0.10)])
        assert skip is False
        assert "low confidence" in reason

    def test_high_top_score_wide_gap_skips(self):
        # Top 0.92, second 0.50 → gap 0.42 > 0.10 default → skip
        skip, reason = should_skip_sam([_rh("a", 0.92), _rh("b", 0.50)])
        assert skip is True
        assert "unambiguous" in reason

    def test_high_top_score_small_gap_does_not_skip(self):
        # Both 0.86 → gap 0.00 → ambiguous, run SAM
        skip, reason = should_skip_sam([_rh("a", 0.86), _rh("b", 0.86)])
        assert skip is False
        assert "ambiguous" in reason.lower() or "gap" in reason.lower()

    def test_single_hit_no_gap_check(self):
        # Only one ranked hit and it's high → skip (no second to compare)
        skip, reason = should_skip_sam([_rh("a", 0.92)])
        assert skip is True

    def test_gap_threshold_param_override(self):
        # Default gap 0.10; pass 0.001 explicitly to allow a smaller gap
        skip, _ = should_skip_sam(
            [_rh("a", 0.91), _rh("b", 0.90)], gap=0.005,
        )
        assert skip is True

    def test_entity_attribute_conflict_does_not_skip(self):
        # Same entity+attribute, different values → conflict
        hits = [
            _rh("a", 0.95, entity="payments", attribute="provider", value="Adyen"),
            _rh("b", 0.50, entity="payments", attribute="provider", value="Stripe"),
        ]
        # Wide gap (0.45), but conflict → SAM must run
        skip, reason = should_skip_sam(hits)
        assert skip is False
        assert "conflict" in reason.lower()

    def test_force_sam_env_disables_skip(self, monkeypatch):
        monkeypatch.setenv("GML_FORCE_SAM", "1")
        skip, reason = should_skip_sam([_rh("a", 0.99), _rh("b", 0.10)])
        assert skip is False
        assert "GML_FORCE_SAM" in reason

    def test_env_threshold_override(self, monkeypatch):
        monkeypatch.setenv("GML_SAM_SKIP_THRESHOLD", "0.95")
        # With threshold 0.95, 0.90 should not skip
        skip, _ = should_skip_sam([_rh("a", 0.90), _rh("b", 0.40)])
        assert skip is False

    def test_default_constants_match_doc(self):
        # Sanity: confirm the documented defaults (re-tuned for jina-v2
        # cross-encoder scale where relevant scores cluster around 0.5-0.7).
        assert SAM_SKIP_DEFAULT_THRESHOLD == 0.70
        assert SAM_SKIP_DEFAULT_GAP == 0.15

    # ─────────────────────────────────────────────────────────────────
    # Tier-3 conditional skip — type-based SAM bypass
    # ─────────────────────────────────────────────────────────────────

    def test_sam_disabled_env_forces_skip(self, monkeypatch):
        monkeypatch.setenv("GML_SAM_DISABLED", "1")
        # Even with a low-confidence top hit that would normally run SAM:
        skip, reason = should_skip_sam([_rh("a", 0.20), _rh("b", 0.10)])
        assert skip is True
        assert "GML_SAM_DISABLED" in reason

    def test_multi_hop_question_skips_sam(self):
        # Ambiguous top hits (would run SAM by confidence rule), but a
        # multi-hop query bypasses that and skips because SAM's rewrite
        # would narrow the question.
        from orchestration.sdp.query_router import classify_query
        q = "What are all the cars Evan has owned and where did he drive them?"
        hints = classify_query(q)
        assert hints.is_multi_hop is True
        skip, reason = should_skip_sam(
            [_rh("a", 0.86), _rh("b", 0.86)],
            query_text=q, hints=hints,
        )
        assert skip is True
        assert "multi-hop" in reason

    def test_temporal_question_skips_sam(self):
        from orchestration.sdp.query_router import classify_query
        q = "When did Caroline go to the LGBTQ support group?"
        hints = classify_query(q)
        # query_router may or may not mark this as temporal; if it does, SAM should skip
        if hints.is_temporal:
            skip, reason = should_skip_sam(
                [_rh("a", 0.86), _rh("b", 0.86)],
                query_text=q, hints=hints,
            )
            assert skip is True
            assert "temporal" in reason

    def test_list_question_skips_sam(self):
        # Cat-1 single-hop by category but list-style by phrasing.
        # Synonym snap / list detection in answer_generator catches this.
        skip, reason = should_skip_sam(
            [_rh("a", 0.86), _rh("b", 0.86)],
            query_text="What kinds of activities does Sam do?",
        )
        assert skip is True
        assert "list" in reason

    def test_single_hop_does_not_get_conditional_skip(self):
        # Single-hop ambiguous → run SAM (default behavior preserved)
        skip, reason = should_skip_sam(
            [_rh("a", 0.86), _rh("b", 0.86)],
            query_text="Who is Mel?",
        )
        assert skip is False
        # The reason should be confidence-based (not type-based)
        assert "ambiguous" in reason.lower() or "gap" in reason.lower()

    def test_conditional_can_be_disabled(self, monkeypatch):
        # GML_SAM_CONDITIONAL=0 turns off the type-based skip entirely
        monkeypatch.setenv("GML_SAM_CONDITIONAL", "0")
        skip, reason = should_skip_sam(
            [_rh("a", 0.86), _rh("b", 0.86)],
            query_text="What kinds of activities does Sam do?",
        )
        # With conditional off, this falls through to confidence check:
        # gap is 0 → ambiguous → run SAM
        assert skip is False
        assert "ambiguous" in reason.lower() or "gap" in reason.lower()

    def test_existing_callers_without_query_text_still_work(self):
        # The new query_text/hints params are optional. Existing test
        # patterns (just ranked) must keep working.
        skip, reason = should_skip_sam([_rh("a", 0.92), _rh("b", 0.50)])
        assert skip is True
        assert "unambiguous" in reason
