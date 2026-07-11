"""Tests for orchestration/sdp/date_extractor.py — Phase B6 date extraction."""
import pytest

from orchestration.sdp.date_extractor import (
    extract_dates,
    extract_entities_with_dates,
    has_date,
)


class TestExtractDates:
    def test_empty(self):
        assert extract_dates("") == []

    def test_no_dates(self):
        assert extract_dates("hello there how are you") == []

    def test_weekday(self):
        out = extract_dates("I went on Sunday to the park")
        assert "sunday" in out

    def test_month_day(self):
        out = extract_dates("My birthday is May 7")
        assert "may 7" in out

    def test_day_month(self):
        out = extract_dates("Born on 7 May")
        # Lower-cased and surface preserved
        assert "7 may" in out

    def test_month_day_year(self):
        out = extract_dates("It happened on May 7, 2023")
        # Match captures full string
        assert any("may 7" in d and "2023" in d for d in out)

    def test_iso_date(self):
        out = extract_dates("Started 2023-05-07")
        assert "2023-05-07" in out

    def test_relative_yesterday(self):
        out = extract_dates("She came yesterday")
        assert "yesterday" in out

    def test_relative_ago(self):
        out = extract_dates("Two weeks ago we moved")
        assert any("ago" in d for d in out)

    def test_multiple_dates_in_one_string(self):
        out = extract_dates("Started Monday, ended Friday, May 7")
        # All three captured
        assert "monday" in out
        assert "friday" in out
        assert any("may 7" in d for d in out)

    def test_dedup(self):
        out = extract_dates("Sunday Sunday SUNDAY sunday")
        # Only one "sunday" in output
        assert out.count("sunday") == 1


class TestHasDate:
    def test_no_date(self):
        assert has_date("hello there") is False

    def test_has_weekday(self):
        assert has_date("we met on Tuesday") is True

    def test_has_month(self):
        assert has_date("born in May") is True

    def test_has_relative(self):
        assert has_date("two days ago") is True

    def test_empty(self):
        assert has_date("") is False


class TestExtractEntitiesWithDates:
    def test_combined(self):
        out = extract_entities_with_dates("Caroline came on Sunday May 7")
        # Both kinds present
        assert "caroline" in out
        assert "sunday" in out
        assert any("may 7" in x for x in out)

    def test_dedup_across_kinds(self):
        # If something matches both name and date regex, only one copy
        out = extract_entities_with_dates("Sunday is a person's name? No, just a day.")
        assert out.count("sunday") == 1
