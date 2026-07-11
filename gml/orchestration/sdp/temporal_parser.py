"""Rule-based natural-language → as_of resolution.

Turns past-anchored phrases in a user query into a concrete ``as_of``
instant so time-travel retrieval works conversationally, not just via the
HTTP parameter. Deliberately rule-based: a dateparser dependency or LLM
call is not worth it for the ~90% of phrasings these patterns cover, and
rules keep the hot path at microseconds.

Two tiers, because valid_from is WORLD time (when the fact began to hold),
not ingest time:

* STRONG calendar anchors — "in March 2025", "in 2024", "on 2025-03-14",
  "3 months ago", "last year". Unambiguous state-reconstruction intent;
  applied by default.
* RELATIVE recency — "yesterday", "last week", "last month". Usually just
  conversational framing ("what did we decide last week" wants the
  decision, which may have been ingested since); excluded by default,
  enabled with GML_NL_AS_OF=all.

GML_NL_AS_OF=0 disables everything. An explicitly supplied as_of always
wins — the parser only fills a None.

Semantics: calendar anchors resolve to the END of the referenced period
("in March 2025" → 2025-03-31 23:59:59 UTC), i.e. the belief state at the
close of that period. Results are clamped to the past; a phrase resolving
to now-or-future returns None.
"""
from __future__ import annotations

import calendar
import os
import re
from datetime import datetime, timedelta, timezone

_MONTHS = {
    name.lower(): i
    for i, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update({
    name.lower(): i
    for i, name in enumerate(calendar.month_abbr)
    if name
})

_MONTH_RE = re.compile(
    r"\bin\s+(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
    r"(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\bin\s+((?:19|20)\d{2})\b")
_ISO_RE = re.compile(r"\bon\s+(\d{4})-(\d{2})-(\d{2})\b")
_AGO_RE = re.compile(
    r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", re.IGNORECASE
)
_LAST_YEAR_RE = re.compile(r"\blast\s+year\b", re.IGNORECASE)
_RELATIVE_RE = re.compile(
    r"\b(yesterday|last\s+week|last\s+month)\b", re.IGNORECASE
)

_AGO_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _mode() -> str:
    return os.environ.get("GML_NL_AS_OF", "strong").strip().lower()


def _end_of_month(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)


def parse_as_of(text: str, now: datetime | None = None) -> datetime | None:
    """Resolve a past-anchored phrase in ``text`` to an as_of instant, or
    None when nothing confidently parses. Never raises."""
    mode = _mode()
    if mode in {"0", "off", "false", "no"}:
        return None
    now = now or datetime.now(timezone.utc)

    candidate: datetime | None = None

    m = _ISO_RE.search(text)
    if m:
        try:
            candidate = datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                23, 59, 59, tzinfo=timezone.utc,
            )
        except ValueError:
            candidate = None

    if candidate is None:
        m = _MONTH_RE.search(text)
        if m:
            month = _MONTHS[m.group(1).lower()]
            if m.group(2):
                year = int(m.group(2))
            else:
                # Bare month: most recent past occurrence.
                year = now.year if month < now.month else now.year - 1
            candidate = _end_of_month(year, month)

    if candidate is None:
        m = _YEAR_RE.search(text)
        if m:
            year = int(m.group(1))
            candidate = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    if candidate is None:
        m = _AGO_RE.search(text)
        if m:
            days = int(m.group(1)) * _AGO_UNIT_DAYS[m.group(2).lower()]
            candidate = now - timedelta(days=days)

    if candidate is None and _LAST_YEAR_RE.search(text):
        candidate = datetime(
            now.year - 1, 12, 31, 23, 59, 59, tzinfo=timezone.utc
        )

    if candidate is None and mode == "all":
        m = _RELATIVE_RE.search(text)
        if m:
            phrase = re.sub(r"\s+", " ", m.group(1).lower())
            days = {"yesterday": 1, "last week": 7, "last month": 30}[phrase]
            candidate = now - timedelta(days=days)

    if candidate is None or candidate >= now:
        return None
    return candidate
