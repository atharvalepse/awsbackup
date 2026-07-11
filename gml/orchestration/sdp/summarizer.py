"""SDP Stage 9 — SemanticSummarizer.

Compresses a clause or short paragraph into a concise summary line.
Heuristic only: pick the first sentence + the clause containing the
extracted value. No LLM.

Tradeoff: this is far inferior to LLM summarization for nuanced content,
but the doc explicitly opts for the lightweight path. SDP's job is fast
ingestion; users who want sentence-fluent rewrites can layer the LLM
extractor on top (see ingest() vs sdp_ingest() in mcp_server.py).
"""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_CHARS = 160


def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class SemanticSummarizer:
    """Pick a representative sentence and trim it."""

    def summarize(self, text: str) -> str:
        if not text:
            return ""
        sentences = _SENT_SPLIT.split(text.strip())
        first = sentences[0].strip() if sentences else text
        return _truncate(first)
