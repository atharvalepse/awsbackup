"""Tests for the JSONL memory store."""
from datetime import datetime, timezone

import pytest

from orchestration.memory_store import JsonlMemoryStore
from orchestration.pipeline.contracts import MemoryItem


def _make_item(id: str, content: str = "...") -> MemoryItem:
    return MemoryItem(
        id=id, content=content,
        timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.5,
    )


def test_store_creates_file_when_missing(tmp_path):
    p = tmp_path / "subdir" / "memories.jsonl"
    store = JsonlMemoryStore(p)
    assert p.exists()
    assert store.load_all() == []


def test_store_round_trip(tmp_path):
    p = tmp_path / "memories.jsonl"
    store = JsonlMemoryStore(p)
    a = _make_item("a", "alpha")
    b = _make_item("b", "beta")
    store.add(a)
    store.add(b)

    loaded = JsonlMemoryStore(p).load_all()
    assert [r.id for r in loaded] == ["a", "b"]
    assert loaded[0].content == "alpha"


def test_store_add_many_appends_all(tmp_path):
    p = tmp_path / "memories.jsonl"
    store = JsonlMemoryStore(p)
    items = [_make_item(f"id-{i}") for i in range(5)]
    store.add_many(items)
    assert len(store.load_all()) == 5


def test_store_skips_invalid_lines(tmp_path):
    p = tmp_path / "memories.jsonl"
    store = JsonlMemoryStore(p)
    store.add(_make_item("ok"))
    # Append a malformed line
    with p.open("a") as f:
        f.write("not json\n")
        f.write('{"id":"missing-required-fields"}\n')
    loaded = store.load_all()
    assert [r.id for r in loaded] == ["ok"]
