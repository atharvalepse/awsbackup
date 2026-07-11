"""SDP Stage 1 — ConversationParser.

Normalizes raw AI conversation turns before semantic extraction. Cleans
whitespace, lowercases roles, parses/defaults timestamps, strips obvious
formatting noise (markdown fences, leading/trailing punctuation).

Input is a list of dicts {role, content, timestamp?, platform?}.
Output is the same shape with normalized fields and a stable schema.
"""
import re
from datetime import datetime, timezone
from typing import Any


_FENCE_RE = re.compile(r"```[\w]*\n?|```")
_MULTI_WS = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Strip code fences, collapse whitespace, trim."""
    if not text:
        return ""
    text = _FENCE_RE.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


def _normalize_timestamp(ts: Any) -> str:
    """Return ISO-8601 UTC. Accepts datetime, ISO string, or None (→ now)."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    if isinstance(ts, str) and ts:
        # Trust strings shaped like ISO; otherwise fall through to now.
        if re.match(r"^\d{4}-\d{2}-\d{2}", ts):
            return ts
    return datetime.now(timezone.utc).isoformat()


class ConversationParser:
    """Normalize raw messages into a canonical list of dicts."""

    def parse(self, messages: list[dict]) -> list[dict]:
        parsed: list[dict] = []
        for msg in messages:
            role = (msg.get("role") or "").strip().lower() or "unknown"
            content = _normalize_text(msg.get("content") or "")
            if not content:
                continue
            parsed.append({
                "role": role,
                "content": content,
                "timestamp": _normalize_timestamp(msg.get("timestamp")),
                "platform": msg.get("platform"),
            })
        return parsed
