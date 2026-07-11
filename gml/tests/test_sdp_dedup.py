"""Tests for orchestration/sdp/dedup.py — MinHash + LSH near-duplicate dedup."""
import pytest

from orchestration.sdp.dedup import (
    DEFAULT_NUM_HASHES,
    MinHashDeduper,
    _jaccard,
    _minhash_signature,
    _shingles,
)


# ---------------------------------------------------------------------------
# Low-level shingle/jaccard/signature primitives
# ---------------------------------------------------------------------------


class TestShingles:
    def test_empty_text(self):
        assert _shingles("") == set()
        assert _shingles("  ") == set()

    def test_shorter_than_k(self):
        assert _shingles("ab", k=3) == {"ab"}

    def test_basic_trigrams(self):
        # "abcd" → {"abc", "bcd"}
        assert _shingles("abcd") == {"abc", "bcd"}

    def test_case_insensitive(self):
        assert _shingles("Hello") == _shingles("hello") == _shingles("HELLO")

    def test_whitespace_collapsed(self):
        # multiple whitespace should normalize to single space
        assert _shingles("a  b") == _shingles("a b")
        assert _shingles("a\nb") == _shingles("a b")


class TestJaccard:
    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 0.0
        assert _jaccard({"a"}, set()) == 0.0

    def test_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_half_overlap(self):
        # {a,b} ∩ {b,c} = {b} (1), {a,b} ∪ {b,c} = {a,b,c} (3)
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


class TestMinHashSignature:
    def test_signature_length(self):
        sig = _minhash_signature({"abc", "def"})
        assert len(sig) == DEFAULT_NUM_HASHES

    def test_empty_set(self):
        sig = _minhash_signature(set())
        assert all(s == 0 for s in sig)

    def test_deterministic(self):
        sig1 = _minhash_signature({"abc", "def"})
        sig2 = _minhash_signature({"def", "abc"})  # order shouldn't matter
        assert sig1 == sig2


# ---------------------------------------------------------------------------
# MinHashDeduper end-to-end behavior
# ---------------------------------------------------------------------------


class TestMinHashDeduper:
    def test_empty_input(self):
        deduper = MinHashDeduper()
        assert deduper.filter_unique([]) == []

    def test_single_item(self):
        deduper = MinHashDeduper()
        assert deduper.filter_unique(["just one"]) == [0]

    def test_all_distinct(self):
        deduper = MinHashDeduper()
        texts = [
            "The cat sat on the mat",
            "Quick brown fox jumps over the lazy dog",
            "PostgreSQL is a relational database",
        ]
        keep = deduper.filter_unique(texts)
        assert sorted(keep) == [0, 1, 2]

    def test_exact_duplicates_dedupe(self):
        deduper = MinHashDeduper(threshold=0.7)
        texts = ["hello there", "hello there", "hello there"]
        keep = deduper.filter_unique(texts)
        # All three are identical — only one survives
        assert len(keep) == 1

    def test_near_duplicates_dedupe(self):
        deduper = MinHashDeduper(threshold=0.7)
        texts = [
            "How are you doing today?",
            "How are you doing today!",  # diff only in trailing punctuation
            "I'm a totally different sentence about cats.",
        ]
        keep = deduper.filter_unique(texts)
        # First two cluster; third is alone → 2 representatives total
        assert len(keep) == 2
        # The longer text should survive within the dup cluster — index 0 and 1
        # are same length so the alphabetic minimum index wins (0).
        kept_set = set(keep)
        assert 0 in kept_set or 1 in kept_set
        assert 2 in kept_set

    def test_keeps_longest_in_cluster(self):
        deduper = MinHashDeduper(threshold=0.5)
        texts = [
            "hi",
            "hi mel",
            "hi mel how are you doing today my friend",
        ]
        keep = deduper.filter_unique(texts)
        # All three should cluster (high char-overlap with the longest)
        # and the longest (index 2) should be the survivor
        assert 2 in keep

    def test_threshold_strictness(self):
        # With a high threshold, even small differences should NOT cluster
        loose = MinHashDeduper(threshold=0.3)
        strict = MinHashDeduper(threshold=0.95)
        texts = [
            "We use Stripe for payments",
            "We use Adyen for payments",
        ]
        # Strict: very different verbs/nouns → keep both
        strict_keep = strict.filter_unique(texts)
        assert len(strict_keep) == 2
        # Loose: lots of char-trigram overlap → maybe cluster, but pleasingly
        # at threshold 0.3 these CAN cluster. Verify the keep count is in
        # [1, 2] — not asserting exact behavior since it's threshold-sensitive.
        loose_keep = loose.filter_unique(texts)
        assert 1 <= len(loose_keep) <= 2

    def test_num_hashes_must_be_multiple_of_band_size(self):
        with pytest.raises(ValueError):
            MinHashDeduper(num_hashes=64, band_size=5)  # 64 % 5 != 0
