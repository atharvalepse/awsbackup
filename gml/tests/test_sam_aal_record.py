"""Tests for orchestration/sam/aal_record.py — AALRecord + AALTuple."""
from datetime import datetime, timezone

import pytest

from orchestration.sam.aal_record import AALRecord, AALTuple


class TestAALTuple:
    def test_to_content_simple(self):
        t = AALTuple(subject="Caroline", verb="research", object="agencies")
        assert t.to_content() == "Caroline research agencies"

    def test_to_content_with_time(self):
        t = AALTuple(subject="X", verb="go", object="Y", time="Sunday")
        assert t.to_content() == "X go Y (Sunday)"

    def test_to_content_negated(self):
        t = AALTuple(subject="X", verb="use", object="Stripe", negated=True)
        assert "did not" in t.to_content()
        assert "Stripe" in t.to_content()

    def test_as_dict_roundtrip(self):
        t = AALTuple(subject="X", verb="go", object="Y", time="Sun", negated=False)
        d = t.as_dict()
        assert d["subject"] == "X"
        assert d["verb"] == "go"
        assert d["time"] == "Sun"


class TestAALRecord:
    def test_empty_default(self):
        r = AALRecord()
        assert r.is_empty is True
        assert r.tuples == []
        assert r.chunk_summary == ""

    def test_non_empty_with_tuples(self):
        r = AALRecord(tuples=[AALTuple("X", "go", "Y")])
        assert r.is_empty is False

    def test_non_empty_with_chunk(self):
        r = AALRecord(chunk_summary="A useful summary")
        assert r.is_empty is False

    def test_chunk_only_whitespace_is_empty(self):
        r = AALRecord(chunk_summary="   ")
        assert r.is_empty is True

    def test_dict_roundtrip(self):
        ts = datetime(2024, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        r = AALRecord(
            tuples=[AALTuple("Caroline", "go", "park", time="Sun")],
            chunk_summary="Caroline went to park",
            entities=["caroline", "sun"],
            timestamp=ts, session_id=5, importance=0.8, confidence=0.95,
        )
        d = r.as_dict()
        r2 = AALRecord.from_dict(d)
        assert r2.chunk_summary == "Caroline went to park"
        assert len(r2.tuples) == 1
        assert r2.tuples[0].subject == "Caroline"
        assert r2.session_id == 5
        assert r2.importance == 0.8

    def test_dict_with_bad_timestamp_falls_back(self):
        d = {"tuples": [], "chunk_summary": "x", "timestamp": "not a date"}
        r = AALRecord.from_dict(d)
        assert isinstance(r.timestamp, datetime)

    def test_from_dict_missing_fields_uses_defaults(self):
        r = AALRecord.from_dict({})
        assert r.tuples == []
        assert r.importance == 0.7
        assert r.confidence == 0.8
