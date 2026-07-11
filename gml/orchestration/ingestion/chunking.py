"""Split over-long memory content into sentence-aligned chunks on insert.

gmlcore normally stores atomic AAL facts (short), so this is a safeguard for
the occasional long blob (e.g. a pasted document). Each chunk becomes its own
MemoryItem sharing a ``parent_memory_id`` so retrieval can deduplicate chunks
that came from the same source. Token count uses a cheap chars/4 heuristic to
avoid pulling a tokenizer onto the write path; override the budget via
``GML_CHUNK_MAX_TOKENS``.
"""
from __future__ import annotations

import os
import re

from orchestration.pipeline.contracts import MemoryItem

# ~chars per token for the chars/4 heuristic.
_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_TOKENS = int(os.environ.get("GML_CHUNK_MAX_TOKENS", "500"))
# Sentence boundary: ., !, ? followed by whitespace. Good enough for prose;
# falls back to a hard character split if a single "sentence" is huge.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def chunk_content(text: str, max_tokens: int = _DEFAULT_MAX_TOKENS) -> list[str]:
    """Split ``text`` into chunks each <= ``max_tokens`` (heuristic), breaking
    at sentence boundaries. Returns ``[text]`` unchanged when it already fits.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    budget = max_tokens * _CHARS_PER_TOKEN
    sentences = _SENTENCE_RE.split(text.strip())
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        # A single sentence longer than the budget: hard-split it.
        while len(s) > budget:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.append(s[:budget].strip())
            s = s[budget:]
        if len(cur) + len(s) + 1 > budget and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip() if cur else s
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c] or [text]


def expand_chunked(
    items: list[MemoryItem], max_tokens: int = _DEFAULT_MAX_TOKENS
) -> list[MemoryItem]:
    """Replace any item whose content exceeds ``max_tokens`` with one item per
    chunk, all sharing ``parent_memory_id`` = the original item's id. Items that
    fit are returned unchanged (so this is a no-op for normal atomic facts).
    """
    out: list[MemoryItem] = []
    for item in items:
        parts = chunk_content(item.content, max_tokens)
        if len(parts) <= 1:
            out.append(item)
            continue
        for i, part in enumerate(parts):
            out.append(
                item.model_copy(
                    update={
                        "id": f"{item.id}#chunk{i}",
                        "content": part,
                        "parent_memory_id": item.id,
                    }
                )
            )
    return out
