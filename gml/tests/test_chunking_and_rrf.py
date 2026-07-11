"""Unit tests for the pure logic added for the recall fixes:

- chunking.chunk_content / expand_chunked  (Fix 6)
- query_expansion.rrf_merge                (Fix 1)
"""
from datetime import datetime, timezone

from orchestration.ingestion.chunking import chunk_content, expand_chunked
from orchestration.pipeline.contracts import MemoryItem, RetrievalHit
from orchestration.retriever.query_expansion import rrf_merge


def _item(rec_id: str, content: str, **kw) -> MemoryItem:
    return MemoryItem(
        id=rec_id, content=content,
        timestamp=datetime.now(timezone.utc),
        source="test", authority_score=0.7, **kw,
    )


def _hit(rec_id: str, sim: float = 0.5) -> RetrievalHit:
    return RetrievalHit(record=_item(rec_id, f"c-{rec_id}"), similarity=sim)


# --------------------------------------------------------------------------
# chunk_content
# --------------------------------------------------------------------------

def test_short_content_is_one_chunk():
    assert chunk_content("I like Rust.", max_tokens=500) == ["I like Rust."]


def test_long_content_splits_under_budget():
    text = "This is a sentence. " * 400  # ~2000 tokens
    chunks = chunk_content(text, max_tokens=100)
    assert len(chunks) > 1
    # chars/4 heuristic: every chunk within the char budget.
    assert all(len(c) // 4 <= 100 for c in chunks)


def test_split_happens_at_sentence_boundaries():
    text = ("Alpha one two three four five. "
            "Beta one two three four five. "
            "Gamma one two three four five.")
    chunks = chunk_content(text, max_tokens=10)  # ~40 chars/chunk
    # No chunk should start mid-sentence (each begins with a capitalized word).
    assert all(c[0].isupper() for c in chunks)
    # Round-trips the words (modulo whitespace), nothing dropped.
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_single_oversized_sentence_is_hard_split():
    text = "word " * 500  # one boundary-less run
    chunks = chunk_content(text, max_tokens=20)  # 80-char budget
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)


def test_empty_content_returns_single():
    assert chunk_content("", max_tokens=500) == [""]


# --------------------------------------------------------------------------
# expand_chunked
# --------------------------------------------------------------------------

def test_atomic_item_unchanged():
    item = _item("m1", "short fact", entity="user", attribute="lang", value="Rust")
    out = expand_chunked([item], max_tokens=500)
    assert out == [item]  # identity-preserving for normal facts


def test_long_item_expands_with_shared_parent():
    item = _item("m1", "This is a sentence. " * 400, entity="doc")
    out = expand_chunked([item], max_tokens=100)
    assert len(out) > 1
    assert all(c.parent_memory_id == "m1" for c in out)
    assert [c.id for c in out[:2]] == ["m1#chunk0", "m1#chunk1"]
    # Non-content fields are carried onto each chunk.
    assert all(c.entity == "doc" and c.source == "test" for c in out)


def test_mixed_batch_only_expands_the_long_one():
    short = _item("s", "tiny")
    long = _item("l", "This is a sentence. " * 400)
    out = expand_chunked([short, long], max_tokens=100)
    assert short in out
    assert sum(1 for c in out if c.parent_memory_id == "l") == len(out) - 1


# --------------------------------------------------------------------------
# rrf_merge
# --------------------------------------------------------------------------

def test_empty_sets_return_empty():
    assert rrf_merge([], k=5) == []
    assert rrf_merge([[], []], k=5) == []


def test_single_set_preserves_order():
    hits = [_hit("a"), _hit("b"), _hit("c")]
    merged = rrf_merge([hits], k=5)
    assert [h.record.id for h in merged] == ["a", "b", "c"]


def test_agreement_across_sets_ranks_higher():
    # B is rank-1 in set1 and rank-0 in set2 -> most votes -> first.
    set1 = [_hit("a"), _hit("b")]
    set2 = [_hit("b"), _hit("c")]
    merged = rrf_merge([set1, set2], k=5)
    assert merged[0].record.id == "b"
    assert {h.record.id for h in merged} == {"a", "b", "c"}  # deduped by id


def test_k_limits_and_scores_normalized():
    merged = rrf_merge([[_hit("a"), _hit("b"), _hit("c")]], k=2)
    assert len(merged) == 2
    assert merged[0].similarity == 1.0  # top normalized to 1.0
    assert all(-1.0 <= h.similarity <= 1.0 for h in merged)
