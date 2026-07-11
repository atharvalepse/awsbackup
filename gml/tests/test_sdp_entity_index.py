"""Tests for orchestration/sdp/entity_index.py — entity hash-index + extraction."""
from datetime import datetime, timezone

import pytest

from orchestration.pipeline.contracts import MemoryItem
from orchestration.sdp.entity_index import (
    EntityIndex,
    extract_entities,
)


def _make_record(rec_id: str, content: str, entity: str | None = None) -> MemoryItem:
    return MemoryItem(
        id=rec_id,
        content=content,
        entity=entity,
        attribute=None,
        value=None,
        timestamp=datetime.now(timezone.utc),
        source="test",
        authority_score=0.7,
        pinned=False,
    )


class TestExtractEntities:
    def test_empty(self):
        assert extract_entities("") == []

    def test_basic_first_name(self):
        ents = extract_entities("Caroline went home")
        assert "caroline" in ents

    def test_stopword_not_extracted(self):
        # "When" at sentence start is a stopword name
        ents = extract_entities("When did this happen?")
        assert "when" not in ents

    def test_first_and_last_name(self):
        ents = extract_entities("Priya Iyer leads the team")
        assert "priya iyer" in ents or "priya" in ents

    def test_multiple_entities(self):
        ents = extract_entities("Caroline and Melanie went to the park")
        assert "caroline" in ents
        assert "melanie" in ents

    def test_normalizes_to_lowercase(self):
        ents = extract_entities("Caroline went home")
        # All output is lowercased
        assert all(e == e.lower() for e in ents)

    def test_dedup_preserves_order(self):
        ents = extract_entities("Caroline saw Caroline at the store")
        assert ents.count("caroline") == 1

    def test_possessive_normalized(self):
        # The regex captures "Caroline" without the apostrophe (\b stops),
        # but if a future regex change preserves it, _normalize_form catches it.
        from orchestration.sdp.entity_index import _normalize_form
        assert _normalize_form("Caroline's") == "caroline"
        assert _normalize_form("caroline’s") == "caroline"
        assert _normalize_form("Kids'") == "kids"

    def test_extract_with_possessive_in_text(self):
        ents = extract_entities("Caroline's friend Melanie called")
        assert "caroline" in ents
        assert "melanie" in ents


class TestEntityIndex:
    def test_empty_lookup(self):
        idx = EntityIndex()
        assert idx.lookup_query("What did Caroline do?") == set()
        assert idx.entity_count == 0

    def test_add_single_record(self):
        idx = EntityIndex()
        rec = _make_record("r1", "Caroline went home", entity="Caroline")
        idx.add(rec)
        assert idx.lookup_query("about Caroline") == {"r1"}
        assert idx.entity_count == 1

    def test_add_extracts_from_content(self):
        idx = EntityIndex()
        # No explicit entity field; should pick up "Melanie" from the text
        rec = _make_record("r1", "Melanie is happy today")
        idx.add(rec)
        assert idx.lookup_query("How is Melanie") == {"r1"}

    def test_lookup_unknown_entity_returns_empty(self):
        idx = EntityIndex()
        idx.add(_make_record("r1", "Caroline went home"))
        assert idx.lookup_query("What about Steve?") == set()

    def test_multiple_records_same_entity(self):
        idx = EntityIndex()
        idx.add(_make_record("r1", "Caroline went home"))
        idx.add(_make_record("r2", "Caroline likes coffee"))
        idx.add(_make_record("r3", "Melanie is busy"))
        result = idx.lookup_query("ask about Caroline")
        assert result == {"r1", "r2"}

    def test_one_record_multiple_entities(self):
        idx = EntityIndex()
        idx.add(_make_record("r1", "Caroline and Melanie went to the park"))
        # r1 should be findable via either entity
        assert "r1" in idx.lookup_query("Caroline?")
        assert "r1" in idx.lookup_query("Melanie?")

    def test_remove(self):
        idx = EntityIndex()
        idx.add(_make_record("r1", "Caroline went home"))
        idx.add(_make_record("r2", "Caroline likes coffee"))
        idx.remove("r1")
        assert idx.lookup_query("Caroline") == {"r2"}
        idx.remove("r2")
        assert idx.lookup_query("Caroline") == set()
        # entity should be cleaned up too
        assert idx.entity_count == 0

    def test_add_many(self):
        idx = EntityIndex()
        idx.add_many([
            _make_record("r1", "Caroline went home"),
            _make_record("r2", "Melanie is happy"),
        ])
        assert idx.record_count == 2

    def test_top_entities(self):
        idx = EntityIndex()
        idx.add_many([
            _make_record("r1", "Caroline went home"),
            _make_record("r2", "Caroline likes coffee"),
            _make_record("r3", "Caroline saw Melanie"),
            _make_record("r4", "Melanie is happy"),
        ])
        top = idx.top_entities(n=2)
        # Caroline appears in r1, r2, r3 (3 records)
        # Melanie appears in r3, r4 (2 records)
        names = [t[0] for t in top]
        assert names[0] == "caroline"

    def test_case_insensitive_lookup(self):
        idx = EntityIndex()
        idx.add(_make_record("r1", "Caroline went home", entity="Caroline"))
        assert idx.lookup_entities(["CAROLINE"]) == {"r1"}
        assert idx.lookup_entities(["caroline"]) == {"r1"}
