"""Date resolver — turn relative-time phrases into absolute ISO dates.

This is the cat-2 lever from the audit. LOCOMO temporal questions expect
absolute-date answers (e.g. "7 May 2023") but messages often say "Sunday"
or "last week". Without resolving, the indexed entity "sunday" never
matches the gold answer "7 May 2023" in the bench scorer.

Given an **anchor** datetime (the session's timestamp, available in the
LOCOMO data as ``session_N_date_time``), this resolver maps relative
phrases like:

    Sunday  →  the previous (or upcoming) Sunday relative to anchor
    yesterday  →  anchor - 1 day
    two weeks ago  →  anchor - 14 days
    last Monday  →  the most recent Monday before anchor
    next month  →  approximated as anchor + 30 days
    May 7  →  the next May 7 (this or next year, whichever's closer)

Pure regex + Python ``datetime`` math. No LLM, no external deps.
Returns ISO date strings (YYYY-MM-DD) so they're stable hash-index keys.

Coverage is intentionally bounded to LOCOMO's vocabulary; we punt on
ambiguous cases ("a while back", "recently") rather than guess.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


WEEKDAY_TO_INT = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTH_TO_INT = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "a": 1, "an": 1,
}

UNIT_TO_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,  # approximate
    "year": 365, "years": 365,
}


_RE_WEEKDAY = re.compile(
    r"\b(?P<mod>last|next|this|previous|coming|past)?\s*(?P<wd>" +
    "|".join(WEEKDAY_TO_INT.keys()) + r")\b", re.IGNORECASE
)
_RE_REL_AGO = re.compile(
    r"\b(?P<num>\d+|" + "|".join(WORD_NUM.keys()) + r")\s+"
    r"(?P<unit>days?|weeks?|months?|years?)\s+ago\b", re.IGNORECASE
)
_RE_LAST_UNIT = re.compile(
    r"\b(?P<mod>last|past|previous)\s+"
    r"(?P<unit>day|week|month|year)\b", re.IGNORECASE
)
_RE_NEXT_UNIT = re.compile(
    r"\b(?P<mod>next|upcoming|coming)\s+"
    r"(?P<unit>day|week|month|year)\b", re.IGNORECASE
)
_RE_SIMPLE = re.compile(
    r"\b(?P<word>yesterday|today|tomorrow)\b", re.IGNORECASE
)
_RE_MONTH_DAY = re.compile(
    r"\b(?P<m>" + "|".join(MONTH_TO_INT.keys()) + r")\s+"
    r"(?P<d>[12]\d|3[01]|0?[1-9])"
    r"(?:[,\s]+(?P<y>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)
_RE_DAY_MONTH = re.compile(
    r"\b(?P<d>[12]\d|3[01]|0?[1-9])\s+"
    r"(?P<m>" + "|".join(MONTH_TO_INT.keys()) + r")"
    r"(?:\s+(?P<y>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)
_RE_ISO = re.compile(r"\b(?P<iso>(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01]))\b")


@dataclass
class ResolvedDate:
    """One resolved absolute date with provenance."""

    iso: str            # "2023-05-07"
    source_phrase: str  # the original surface form, lowercased
    confidence: float = 1.0


def _parse_anchor(anchor: datetime | str | None) -> datetime | None:
    if anchor is None:
        return None
    if isinstance(anchor, datetime):
        return anchor if anchor.tzinfo else anchor.replace(tzinfo=timezone.utc)
    if isinstance(anchor, str):
        try:
            dt = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None
    return None


def _weekday_resolve(anchor: datetime, weekday_int: int, modifier: str) -> datetime:
    """Resolve "Sunday" / "last Sunday" / "next Sunday" relative to anchor."""
    cur = anchor.weekday()
    mod = (modifier or "").lower()
    if mod == "next" or mod == "coming":
        # Strictly forward — the upcoming weekday
        delta = (weekday_int - cur) % 7
        if delta == 0:
            delta = 7
        return anchor + timedelta(days=delta)
    if mod in ("last", "previous", "past"):
        # Strictly backward — the previous weekday
        delta = (cur - weekday_int) % 7
        if delta == 0:
            delta = 7
        return anchor - timedelta(days=delta)
    # No modifier or "this": pick the closer past or future, prefer past
    # (LOCOMO tends to talk about things that have happened).
    delta_back = (cur - weekday_int) % 7
    if delta_back == 0:
        return anchor  # same weekday as anchor
    return anchor - timedelta(days=delta_back)


def _num_from(token: str) -> int:
    if token.isdigit():
        return int(token)
    return WORD_NUM.get(token.lower(), 1)


def _month_int(name: str) -> int:
    return MONTH_TO_INT.get(name.lower(), 0)


def resolve_dates(text: str, anchor: datetime | str | None = None) -> list[ResolvedDate]:
    """Resolve all relative + absolute dates in ``text`` to ISO strings.

    Returns one :class:`ResolvedDate` per surface form found. Empty list
    when no dates resolvable, OR when no anchor is provided AND the text
    only contains relative forms.

    Each ``iso`` field is "YYYY-MM-DD". Confidence is 1.0 for explicit
    absolute dates, ~0.9 for weekday/relative when anchor is known, 0.5
    for very approximate ("next month").
    """
    if not text:
        return []
    anc = _parse_anchor(anchor)
    out: list[ResolvedDate] = []
    seen: set[str] = set()

    def _emit(iso: str, source: str, conf: float) -> None:
        key = (iso, source.lower())
        if key in seen:
            return
        seen.add(key)
        out.append(ResolvedDate(iso=iso, source_phrase=source.lower(), confidence=conf))

    # ISO dates first (most specific)
    for m in _RE_ISO.finditer(text):
        iso = m.group("iso")
        try:
            dt = datetime.fromisoformat(iso)
            _emit(dt.strftime("%Y-%m-%d"), iso, 1.0)
        except ValueError:
            continue

    # Absolute "May 7" / "May 7, 2023" / "May 7 2023"
    for m in _RE_MONTH_DAY.finditer(text):
        month = _month_int(m.group("m"))
        if not month:
            continue
        try:
            day = int(m.group("d"))
        except (TypeError, ValueError):
            continue
        year_grp = m.group("y")
        if year_grp:
            year = int(year_grp)
        elif anc:
            # No year: assume same year as anchor; if that's in the future,
            # roll back one year (LOCOMO tends to discuss past events).
            year = anc.year
            try:
                candidate = datetime(year, month, day, tzinfo=anc.tzinfo)
                if candidate > anc + timedelta(days=60):
                    year -= 1
            except ValueError:
                continue
        else:
            continue  # no anchor and no year — can't resolve
        try:
            dt = datetime(year, month, day, tzinfo=(anc.tzinfo if anc else timezone.utc))
        except ValueError:
            continue
        _emit(dt.strftime("%Y-%m-%d"), m.group(0), 0.98)

    # "7 May" / "7 May 2023"
    for m in _RE_DAY_MONTH.finditer(text):
        month = _month_int(m.group("m"))
        if not month:
            continue
        try:
            day = int(m.group("d"))
        except (TypeError, ValueError):
            continue
        year_grp = m.group("y")
        if year_grp:
            year = int(year_grp)
        elif anc:
            year = anc.year
            try:
                candidate = datetime(year, month, day, tzinfo=anc.tzinfo)
                if candidate > anc + timedelta(days=60):
                    year -= 1
            except ValueError:
                continue
        else:
            continue
        try:
            dt = datetime(year, month, day, tzinfo=(anc.tzinfo if anc else timezone.utc))
        except ValueError:
            continue
        _emit(dt.strftime("%Y-%m-%d"), m.group(0), 0.98)

    if anc is None:
        return out  # rest of patterns need an anchor

    # "yesterday", "today", "tomorrow"
    for m in _RE_SIMPLE.finditer(text):
        word = m.group("word").lower()
        delta = {"yesterday": -1, "today": 0, "tomorrow": 1}[word]
        dt = anc + timedelta(days=delta)
        _emit(dt.strftime("%Y-%m-%d"), word, 0.95)

    # "two weeks ago", "five days ago"
    for m in _RE_REL_AGO.finditer(text):
        num = _num_from(m.group("num"))
        unit = m.group("unit").lower().rstrip("s") + ("s" if not m.group("unit").lower().endswith("s") else "")
        unit_key = m.group("unit").lower()
        days = UNIT_TO_DAYS.get(unit_key, UNIT_TO_DAYS.get(unit_key.rstrip("s"), 1))
        dt = anc - timedelta(days=num * days)
        _emit(dt.strftime("%Y-%m-%d"), m.group(0), 0.9)

    # "last week" / "next month" / "this year" etc.
    for m in _RE_LAST_UNIT.finditer(text):
        unit_key = m.group("unit").lower()
        days = UNIT_TO_DAYS.get(unit_key, 1)
        dt = anc - timedelta(days=days)
        _emit(dt.strftime("%Y-%m-%d"), m.group(0), 0.6)

    for m in _RE_NEXT_UNIT.finditer(text):
        unit_key = m.group("unit").lower()
        days = UNIT_TO_DAYS.get(unit_key, 1)
        dt = anc + timedelta(days=days)
        _emit(dt.strftime("%Y-%m-%d"), m.group(0), 0.6)

    # Weekday names (handle "last Sunday", "next Friday", bare "Sunday")
    for m in _RE_WEEKDAY.finditer(text):
        wd = WEEKDAY_TO_INT.get(m.group("wd").lower())
        if wd is None:
            continue
        modifier = m.group("mod") or ""
        dt = _weekday_resolve(anc, wd, modifier)
        # Skip ridiculous results (e.g. way before LOCOMO start)
        if abs((dt - anc).days) > 365:
            continue
        _emit(dt.strftime("%Y-%m-%d"), m.group(0), 0.85 if modifier else 0.75)

    return out


def resolve_to_iso_set(text: str, anchor: datetime | str | None = None) -> set[str]:
    """Convenience: return just the set of ISO date strings present in ``text``."""
    return {r.iso for r in resolve_dates(text, anchor)}


def enrich_with_resolved_dates(
    text: str, anchor: datetime | str | None = None,
    max_annotations: int = 3,
) -> str:
    """Append resolved-date annotations to ``text``.

    For each relative date in the text (e.g. "Sunday", "yesterday"), append
    its resolved ISO form AND a human-readable canonical form. This:

    1. Makes the resolved date visible in embeddings — bge-large will see
       both "Sunday" and "2023-05-07" tokens.
    2. Puts the resolved date in ``formatted_context`` so the bench's
       ``context_recall`` metric (which checks gold-answer-in-context)
       can match against LOCOMO golds like "7 May 2023".
    3. Adds ISO surface forms to ``extract_dates`` so EntityIndex
       indexes the resolved dates automatically.

    Skips dates already present in the text in any of the candidate forms,
    so no spurious duplication. Capped at ``max_annotations`` distinct
    resolutions to keep content size bounded.

    Returns the original text unchanged when no resolutions are produced.
    """
    if not text or anchor is None:
        return text
    resolved = resolve_dates(text, anchor)
    if not resolved:
        return text

    text_lower = text.lower()
    annotations: list[str] = []
    seen_isos: set[str] = set()
    for r in resolved:
        if r.iso in seen_isos:
            continue
        # Skip if ISO is already textually present
        if r.iso in text:
            continue
        # Build readable canonical form ("Sunday May 7 2023")
        try:
            dt = datetime.fromisoformat(r.iso)
            readable = dt.strftime("%A %B %-d %Y")
        except (ValueError, OSError):
            # %-d isn't portable on Windows; fall back
            try:
                dt = datetime.fromisoformat(r.iso)
                readable = dt.strftime("%A %B %d %Y").replace(" 0", " ")
            except ValueError:
                readable = r.iso
        # If the readable form is already in the text (case-insensitive),
        # only add the ISO part.
        if readable.lower() in text_lower:
            annotations.append(r.iso)
        else:
            annotations.append(f"{r.iso} ({readable})")
        seen_isos.add(r.iso)
        if len(annotations) >= max_annotations:
            break

    if not annotations:
        return text
    return f"{text} [resolved: {', '.join(annotations)}]"
