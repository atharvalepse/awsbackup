"""Date / time entity extractor — Phase B6.

Why this exists
---------------
LOCOMO multi-hop temporal questions (cat 2, scored 0.13 on baseline)
need the system to retrieve memories that pair an EVENT with a DATE.
Example:

    Q: "When did Caroline go to the LGBTQ support group?"
    A: "7 May 2023"
    Evidence: D1:3 → "Caroline: I went on Sunday, May 7"

Embedding alone misses these because the question doesn't lexically
mention "Sunday" or "May 7" — it asks "when". The fix: at ingest time,
extract every date-like surface form from each memory and index it
under the EntityIndex alongside people. At query time, if the question
asks "when", lookup_query returns all date-bearing memories — which
narrows the candidate pool to where the answer lives.

This module exports two things:
  * ``extract_dates(text) → list[str]`` — pure regex extraction, no LLM.
  * Date patterns are folded into ``EntityIndex.add()`` automatically when
    you import-and-use ``extract_entities_with_dates`` instead of
    ``extract_entities``.

Coverage (LOCOMO-tuned):
  * weekdays:        Monday, Tuesday, ..., Sun
  * months:          January .. December, abbreviated (Jan, Feb, ...)
  * absolute dates:  May 7, 7 May, May 7 2023, 7/5/2023, 2023-05-07
  * relative:        yesterday, today, tomorrow, last week, next month,
                     two days ago, three weeks ago
"""
import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_WEEKDAYS = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTHS = (
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December)"
)
_YEAR = r"(?:19|20)\d{2}"
_DAY = r"(?:[12]\d|3[01]|0?[1-9])"  # 1-31

# Absolute "May 7", "7 May", "May 7 2023", "May 7, 2023"
_ABS_MONTH_DAY = re.compile(
    rf"\b{_MONTHS}\s+{_DAY}(?:\s*,?\s*{_YEAR})?\b"
)
_ABS_DAY_MONTH = re.compile(
    rf"\b{_DAY}\s+{_MONTHS}(?:\s+{_YEAR})?\b"
)
# YYYY-MM-DD
_ISO_DATE = re.compile(rf"\b{_YEAR}-(?:0?[1-9]|1[0-2])-{_DAY}\b")
# DD/MM/YYYY or MM/DD/YYYY — ambiguous, surface as-is
_SLASH_DATE = re.compile(rf"\b{_DAY}/(?:0?[1-9]|1[0-2])/{_YEAR}\b")

# Weekdays (Sunday, Mon, etc.)
_WEEKDAY_RE = re.compile(rf"\b{_WEEKDAYS}\b")

# Standalone month names (rare on their own, but possible)
_MONTH_RE = re.compile(rf"\b{_MONTHS}\b")

# Relative time expressions
_RELATIVE_RE = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|"
    r"last (?:week|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"next (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this (?:morning|afternoon|evening|week|month|year)|"
    r"(?:a|one|two|three|four|five|six|seven|eight|nine|ten|\d+) (?:day|week|month|year|hour|minute)s? ago"
    r")\b",
    re.IGNORECASE,
)

# Generic time-of-day
_TIME_OF_DAY_RE = re.compile(
    r"\b(?:morning|afternoon|evening|night|noon|midnight)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------


def extract_dates(text: str) -> list[str]:
    """Extract all date-like surface forms from ``text``.

    Returns a list of lowercased, deduplicated date strings preserving
    the order of first appearance. Empty list when no dates found.

    No date PARSING happens here — we return the surface form, lowercased.
    A separate stage can normalize (e.g. "Sunday" → next-occurring weekday).
    """
    if not text:
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        n = s.strip().lower()
        if n and n not in seen:
            seen.add(n)
            found.append(n)

    # Order matters — try MOST specific patterns first so they capture
    # whole-date strings before the simpler standalone patterns claim parts.
    for m in _ABS_MONTH_DAY.finditer(text):
        _add(m.group(0))
    for m in _ABS_DAY_MONTH.finditer(text):
        _add(m.group(0))
    for m in _ISO_DATE.finditer(text):
        _add(m.group(0))
    for m in _SLASH_DATE.finditer(text):
        _add(m.group(0))
    for m in _RELATIVE_RE.finditer(text):
        _add(m.group(0))
    for m in _WEEKDAY_RE.finditer(text):
        _add(m.group(0))
    for m in _MONTH_RE.finditer(text):
        _add(m.group(0))
    for m in _TIME_OF_DAY_RE.finditer(text):
        _add(m.group(0))

    return found


# ---------------------------------------------------------------------------


def has_date(text: str) -> bool:
    """Cheap predicate — does ``text`` contain at least one date-like form?

    Used by the answer-grounded scorer (Phase B8): for "when"-style
    questions, candidates with at least one date pattern get a similarity
    boost.
    """
    if not text:
        return False
    for rx in (
        _ABS_MONTH_DAY, _ABS_DAY_MONTH, _ISO_DATE, _SLASH_DATE,
        _RELATIVE_RE, _WEEKDAY_RE, _MONTH_RE,
    ):
        if rx.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Combined entity extraction — people + dates, both lowercased
# ---------------------------------------------------------------------------


def extract_entities_with_dates(text: str) -> list[str]:
    """Convenience: union of ``extract_entities`` (people) + ``extract_dates``.

    Used by the EntityIndex so a single ``add()`` call covers both kinds.
    """
    # Local import to avoid a cycle: entity_index doesn't import this module
    from orchestration.sdp.entity_index import extract_entities

    out: list[str] = []
    seen: set[str] = set()
    for e in extract_entities(text):
        if e not in seen:
            seen.add(e)
            out.append(e)
    for d in extract_dates(text):
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out
