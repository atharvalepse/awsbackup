"""Tests for the heuristic entity synthesizer."""
from datetime import datetime, timezone, timedelta

import pytest

from orchestration.pipeline.contracts import MemoryItem
from orchestration.sdp.entity_synth import synthesize_entity_memories


def _mem(content: str, ts_days_ago: int = 0, mid: str = "m") -> MemoryItem:
    return MemoryItem(
        id=f"{mid}-{ts_days_ago}",
        content=content,
        timestamp=datetime.now(timezone.utc) - timedelta(days=ts_days_ago),
        source="locomo-raw",
        authority_score=0.7,
        pinned=False,
    )


class TestEntitySynth:
    def test_emits_synth_for_frequent_entity(self):
        mems = [
            _mem("Sam took up painting in May", 30),
            _mem("Sam goes kayaking on weekends", 25),
            _mem("Sam hikes weekly with the dog", 20),
            _mem("Sam cooks new Indian recipes", 15),
            _mem("Sam runs every morning at 6 am", 10),
            _mem("Caroline mentioned a yoga class", 5),  # only 1 Caroline ref
        ]
        synths = synthesize_entity_memories(
            mems, top_entities=[("Sam", 5), ("Caroline", 1)],
        )
        assert len(synths) == 1
        s = synths[0]
        assert s.source == "aal-entity-synth"
        assert s.entity == "Sam"
        assert s.authority_score == 0.78
        # All 5 Sam snippets land
        c = s.content.lower()
        for term in ["painting", "kayaking", "hikes", "cooks", "runs"]:
            assert term in c, f"missing {term} in synth content"
        assert s.id.startswith("synth-")

    def test_skips_entities_below_min_memories(self):
        mems = [
            _mem("Mel went to Tokyo", 10),
            _mem("Mel returned Tuesday", 5),
        ]
        # Mel appears twice; min=3 → skip
        synths = synthesize_entity_memories(
            mems, top_entities=["Mel"], min_memories=3,
        )
        assert synths == []

    def test_dedupes_near_duplicate_snippets(self):
        # Two identical-prefix memories — only one should survive in snippets
        mems = [
            _mem("Caroline went to yoga class on Thursday morning", 5),
            _mem("Caroline went to yoga class on Thursday morning", 4),
            _mem("Caroline started meditation on Monday", 3),
            _mem("Caroline finished her certification on Friday", 2),
        ]
        synths = synthesize_entity_memories(
            mems, top_entities=["Caroline"], min_memories=3,
        )
        assert len(synths) == 1
        # Yoga line should appear once, not twice
        c = synths[0].content
        assert c.lower().count("went to yoga") == 1

    def test_handles_tuple_input(self):
        mems = [_mem(f"Caroline mention {i}", i) for i in range(5)]
        # The entity_index.top_entities() returns list[tuple[str, int]]
        synths = synthesize_entity_memories(
            mems, top_entities=[("Caroline", 5)],
        )
        assert len(synths) == 1
        assert synths[0].entity == "Caroline"

    def test_handles_str_input(self):
        mems = [_mem(f"Evan mention {i}", i) for i in range(5)]
        synths = synthesize_entity_memories(
            mems, top_entities=["Evan"],
        )
        assert len(synths) == 1
        assert synths[0].entity == "Evan"

    def test_respects_max_snippets(self):
        # 20 memories about Sam — synth should cap at max_snippets
        mems = [_mem(f"Sam did activity number {i} today", i) for i in range(20)]
        synths = synthesize_entity_memories(
            mems, top_entities=["Sam"], max_snippets=5,
        )
        assert len(synths) == 1
        lines = [ln for ln in synths[0].content.split("\n") if ln.startswith("- ")]
        assert len(lines) == 5

    def test_empty_inputs(self):
        assert synthesize_entity_memories([], top_entities=["Sam"]) == []
        assert synthesize_entity_memories([_mem("x")], top_entities=[]) == []

    def test_id_is_stable(self):
        # Same entity → same synth id across calls (idempotent ingest)
        mems = [_mem(f"Sam {i}", i) for i in range(5)]
        a = synthesize_entity_memories(mems, top_entities=["Sam"])
        b = synthesize_entity_memories(mems, top_entities=["Sam"])
        assert a[0].id == b[0].id

    def test_entity_case_insensitive_match(self):
        mems = [
            _mem("SAM went to the store", 3),
            _mem("Sam called his mom", 2),
            _mem("sam ate breakfast", 1),
        ]
        synths = synthesize_entity_memories(
            mems, top_entities=["Sam"],
        )
        assert len(synths) == 1
        # All three should be matched despite case differences
        lines = [ln for ln in synths[0].content.split("\n") if ln.startswith("- ")]
        assert len(lines) == 3

    def test_skips_too_short_entity_names(self):
        mems = [_mem("a a a", i) for i in range(5)]
        # 1-char entity name should be skipped
        synths = synthesize_entity_memories(mems, top_entities=["a"])
        assert synths == []
