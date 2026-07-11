"""Entity hash-index — O(1) entity → memory_ids lookup.

Purpose
-------
For paraphrase-heavy questions like *"What did Caroline research?"*,
embedding cosine often returns the wrong message (the verb 'research'
doesn't lexically match the message's 'looking into'). But if we KNOW
the question mentions the entity "Caroline", we can:

  1. Look up all memory_ids that mention Caroline   (hash lookup)
  2. Restrict the embedding retrieval to that subset
  3. The right memory's rank jumps from #7 to #1

This is the hash-indexing the doc asked about: a Python ``dict[str, set[str]]``
maintained alongside the retriever. O(1) inserts, O(1) lookups, fits in
memory at LOCOMO scale (and well beyond).

Design
------
- Entities are normalized (lower, strip) and stored as case-insensitive keys.
- Each MemoryItem can be indexed by multiple entities (a window containing
  three names indexes under each).
- ``lookup_query(text)`` extracts likely entities from a question via
  regex + capitalized-name pattern and returns matching memory_ids.
- Designed to plug in BEFORE the retriever as a candidate pre-filter:
  if entity hits exist, retriever ranks within that subset; if not,
  retriever falls back to full search.

The class is intentionally NOT async — all operations are pure in-process
dict ops. Wrap retrieval calls separately.
"""
import re
from collections import defaultdict
from typing import Iterable

from orchestration.pipeline.contracts import MemoryItem


# Captures sequences like "Caroline", "Mel", "Priya Iyer" — capitalized
# tokens that aren't sentence-initial. Tuned for LOCOMO-style first-name
# heavy text. The lookahead and lookbehind keep it from grabbing the
# first word of a sentence.
_CAP_NAME_RE = re.compile(
    r"(?<![.!?])\b([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)\b"
)
# Catch leading capitalized names too (sentence start) — only if NOT
# followed by lowercase common-word punctuation patterns.
_LEADING_NAME_RE = re.compile(r"^([A-Z][a-z]{1,15})\b")

# Stopword names — capitalized words that aren't actually entities.
_NAME_STOPWORDS = {
    "the", "a", "an", "i", "we", "they", "you", "he", "she", "it",
    "yes", "no", "ok", "hey", "hi", "hello", "what", "when", "where",
    "who", "why", "how", "did", "does", "do", "is", "are", "was", "were",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _normalize_form(name: str) -> str:
    """Normalize plural / possessive surface forms to a canonical key.

    Rules:
      - Trailing "'s" or "'s" → drop ("Caroline's" → "Caroline")
      - Trailing "s'" → drop ("kids'" → "kids" — but actually "kid"; we
        keep "kids" since we don't know if it's plural or possessive-plural)
      - Trailing single "s" is RISKY (e.g. "James" → "Jame") so we only
        strip it when the result would still match a known name and avoid
        common single-syllable name endings.
    """
    n = name.strip().lower()
    # Possessive forms
    for suffix in ("'s", "’s"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    # Plural-possessive (kids' → kids)
    if n.endswith("'") or n.endswith("’"):
        n = n[:-1]
    return n


def extract_entities(text: str) -> list[str]:
    """Best-effort entity surface extraction.

    Returns lowercased entity strings, with plural/possessive forms
    normalized ("Caroline's" → "caroline", "Carolines'" → "carolines").
    Designed for LOCOMO's first-name + occasional last-name domain.
    """
    if not text:
        return []
    found: list[str] = []
    for m in _CAP_NAME_RE.finditer(text):
        name = m.group(1).strip()
        norm = _normalize_form(name)
        if norm in _NAME_STOPWORDS or len(norm) < 2:
            continue
        found.append(norm)
    # Also try the sentence-leading form
    lead = _LEADING_NAME_RE.match(text)
    if lead:
        norm = _normalize_form(lead.group(1).strip())
        if norm and norm not in _NAME_STOPWORDS and len(norm) >= 2:
            found.append(norm)
    # Dedup while preserving order
    seen, out = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _entities_for_record(rec: MemoryItem) -> set[str]:
    """Extract entities (people + dates) a MemoryItem should be indexed under.

    Names come from ``extract_entities``; date surface forms come from
    ``extract_dates`` (added in Phase B6). Indexing dates alongside
    people lets temporal questions like "when did X happen?" filter to
    date-bearing memories instantly.
    """
    # Local import to avoid the import cycle (date_extractor imports us).
    from orchestration.sdp.date_extractor import extract_dates

    ents: set[str] = set()
    if rec.entity:
        ents.add(rec.entity.lower())
    for e in extract_entities(rec.content):
        ents.add(e)
    for d in extract_dates(rec.content):
        ents.add(d)
    return ents


class EntityIndex:
    """In-memory inverted index from entity surface form → memory_ids."""

    def __init__(self) -> None:
        self._by_entity: dict[str, set[str]] = defaultdict(set)
        # Reverse mapping so we can clean up on remove
        self._by_id: dict[str, set[str]] = defaultdict(set)

    # ---- mutation -----------------------------------------------------

    def add(self, record: MemoryItem) -> None:
        ents = _entities_for_record(record)
        for ent in ents:
            self._by_entity[ent].add(record.id)
            self._by_id[record.id].add(ent)

    def add_many(self, records: Iterable[MemoryItem]) -> None:
        for r in records:
            self.add(r)

    def remove(self, memory_id: str) -> None:
        ents = self._by_id.pop(memory_id, set())
        for ent in ents:
            self._by_entity.get(ent, set()).discard(memory_id)
            if not self._by_entity.get(ent):
                self._by_entity.pop(ent, None)

    # ---- query --------------------------------------------------------

    def lookup_entities(self, entities: Iterable[str]) -> set[str]:
        """Union of memory_ids matching any of the given entity strings."""
        out: set[str] = set()
        for e in entities:
            key = e.lower().strip()
            if key in self._by_entity:
                out |= self._by_entity[key]
        return out

    def lookup_query(self, query_text: str) -> set[str]:
        """Extract people + dates from query_text and return matching memory_ids.

        Both kinds of entities go through the same hash-lookup. Dates were
        added in Phase B6 to support multi-hop temporal questions.

        Empty set means no entity match — retrieval should fall back to
        its normal (unfiltered) path.
        """
        # Combine people + dates for symmetry with how indexing works.
        from orchestration.sdp.date_extractor import extract_dates
        entities = list(extract_entities(query_text)) + list(extract_dates(query_text))
        return self.lookup_entities(entities)

    # ---- introspection -----------------------------------------------

    def __len__(self) -> int:
        return len(self._by_entity)

    @property
    def entity_count(self) -> int:
        return len(self._by_entity)

    @property
    def record_count(self) -> int:
        return len(self._by_id)

    def top_entities(self, n: int = 10) -> list[tuple[str, int]]:
        """Most-indexed entities and their memory counts."""
        return sorted(
            ((e, len(ids)) for e, ids in self._by_entity.items()),
            key=lambda x: -x[1],
        )[:n]
