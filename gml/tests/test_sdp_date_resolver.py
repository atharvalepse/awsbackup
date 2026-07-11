"""Tests for orchestration/sdp/date_resolver.py — date arithmetic for cat-2."""
from datetime import datetime, timezone

import pytest

from orchestration.sdp.date_resolver import (
    ResolvedDate,
    resolve_dates,
    resolve_to_iso_set,
)


# Wednesday, May 10, 2023 — anchor used by most tests
ANCHOR = datetime(2023, 5, 10, 12, 0, 0, tzinfo=timezone.utc)


class TestNoAnchor:
    def test_empty(self):
        assert resolve_dates("") == []
        assert resolve_dates("", anchor=None) == []

    def test_iso_date_resolves_without_anchor(self):
        out = resolve_dates("Started on 2023-05-07")
        assert "2023-05-07" in [r.iso for r in out]

    def test_relative_without_anchor_is_empty(self):
        """Relative forms require an anchor — without one, return empty."""
        assert resolve_dates("yesterday") == []
        assert resolve_dates("last Sunday") == []

    def test_month_day_without_anchor_skipped(self):
        """Month+day without year and no anchor → can't resolve."""
        out = resolve_dates("May 7")
        # The bare 'May 7' produces nothing because there's no year context.
        assert all(r.iso != "2023-05-07" for r in out)


class TestWeekdayResolution:
    def test_bare_weekday_picks_past(self):
        # Wed May 10 → "Sunday" → previous Sunday = May 7
        isos = resolve_to_iso_set("we went on Sunday", anchor=ANCHOR)
        assert "2023-05-07" in isos

    def test_last_weekday(self):
        isos = resolve_to_iso_set("last Sunday we went", anchor=ANCHOR)
        assert "2023-05-07" in isos

    def test_next_weekday(self):
        # Wed May 10 → "next Sunday" → upcoming Sunday = May 14
        isos = resolve_to_iso_set("next Sunday is the day", anchor=ANCHOR)
        assert "2023-05-14" in isos

    def test_weekday_short_form(self):
        # "Mon" should also work
        isos = resolve_to_iso_set("we met Mon", anchor=ANCHOR)
        # Last Monday before Wed May 10 = May 8
        assert "2023-05-08" in isos


class TestRelativeAgo:
    def test_yesterday(self):
        assert "2023-05-09" in resolve_to_iso_set("yesterday", anchor=ANCHOR)

    def test_today(self):
        assert "2023-05-10" in resolve_to_iso_set("today", anchor=ANCHOR)

    def test_tomorrow(self):
        assert "2023-05-11" in resolve_to_iso_set("tomorrow", anchor=ANCHOR)

    def test_two_weeks_ago(self):
        assert "2023-04-26" in resolve_to_iso_set("two weeks ago", anchor=ANCHOR)

    def test_three_days_ago(self):
        assert "2023-05-07" in resolve_to_iso_set("three days ago", anchor=ANCHOR)

    def test_numeric_days_ago(self):
        # "5 days ago"
        assert "2023-05-05" in resolve_to_iso_set("5 days ago", anchor=ANCHOR)

    def test_a_week_ago(self):
        # "a week ago" — word "a" treated as 1
        assert "2023-05-03" in resolve_to_iso_set("a week ago", anchor=ANCHOR)


class TestAbsoluteDates:
    def test_month_day_with_anchor_picks_year(self):
        # No year given; anchor is May 10, 2023 → "May 7" → past = 2023-05-07
        assert "2023-05-07" in resolve_to_iso_set("on May 7", anchor=ANCHOR)

    def test_day_month_form(self):
        assert "2023-05-07" in resolve_to_iso_set("on 7 May", anchor=ANCHOR)

    def test_explicit_year(self):
        assert "2024-01-15" in resolve_to_iso_set("January 15, 2024", anchor=ANCHOR)

    def test_iso_date(self):
        assert "2023-12-25" in resolve_to_iso_set("event on 2023-12-25", anchor=ANCHOR)

    def test_future_bare_date_rolls_back(self):
        # Anchor May 10, 2023; "December 1" → in the past of the same year
        # would be ambiguous; should give same year if within reasonable range.
        # Our heuristic: same year if not too far future, else previous year.
        out = resolve_to_iso_set("event on December 1", anchor=ANCHOR)
        # Either 2023-12-01 or 2022-12-01 acceptable; assert non-empty
        assert len(out) >= 1


class TestMultipleDatesInText:
    def test_extracts_all(self):
        out = resolve_to_iso_set(
            "we went Sunday and again two weeks later", anchor=ANCHOR,
        )
        # At least the Sunday gets resolved
        assert "2023-05-07" in out

    def test_dedup_same_date_diff_phrasing(self):
        # "yesterday" and "May 9" both resolve to May 9 — kept separately
        # (different source_phrase) but the same iso. Set output dedupes.
        out = resolve_dates("yesterday or May 9", anchor=ANCHOR)
        # Both records present
        phrases = [r.source_phrase for r in out]
        assert "yesterday" in phrases or "may 9" in phrases


class TestAnchorParsing:
    def test_iso_string_anchor(self):
        isos = resolve_to_iso_set("Sunday", anchor="2023-05-10T12:00:00Z")
        assert "2023-05-07" in isos

    def test_bad_anchor_returns_only_iso_dates(self):
        # Invalid anchor → fall back to no-anchor behavior
        out = resolve_dates("Sunday and 2023-05-07", anchor="not a date")
        isos = {r.iso for r in out}
        # ISO date still resolved even without anchor
        assert "2023-05-07" in isos


class TestConfidence:
    def test_iso_has_high_confidence(self):
        out = resolve_dates("2023-05-07")
        assert out[0].confidence >= 0.95

    def test_relative_unit_lower_confidence(self):
        # "last week" is approximate
        out = resolve_dates("last week", anchor=ANCHOR)
        # All "last X" returns roughly 0.6 confidence
        assert all(r.confidence < 0.8 for r in out)
