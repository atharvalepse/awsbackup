"""Title + summary generation for a captured chat turn.

Produces the headline + synopsis shown on a conversation memory card. Reuses
the same :class:`OllamaClient` SAM/the extractor use (default deepseek-r1:8b),
so there's no new model dependency. Fails soft to a deterministic local title
(like :meth:`MemoryExtractor.extract` returning ``[]`` on LLM failure) so a
card always gets sensible text.
"""
from __future__ import annotations

import json
import re

from orchestration.observability.logging import StructuredLogger
from orchestration.sam._ollama_client import OllamaClient

slog = StructuredLogger("summarizer")

_SYSTEM = (
    "You title and summarize a single AI chat turn for a personal memory "
    "dashboard. Given the user's message and the assistant's reply, return "
    "ONLY a JSON object: {\"title\": \"...\", \"summary\": \"...\"}.\n"
    "- title: a concise, specific headline naming the actual topic "
    "(max ~10 words), no surrounding quotes.\n"
    "- summary: 2-3 objective, information-dense sentences capturing what was "
    "asked and the key takeaways of the reply.\n"
    "No preamble, no markdown, no mention of being an AI. JSON only."
)


def _build_prompt(user_prompt: str, ai_response: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        f"USER:\n{(user_prompt or '').strip()[:4000]}\n\n"
        f"ASSISTANT:\n{(ai_response or '').strip()[:6000]}"
    )


def _parse(answer: str) -> dict | None:
    if not answer:
        return None
    text = answer.strip()
    open_idx, close_idx = text.find("{"), text.rfind("}")
    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        return None
    try:
        obj = json.loads(text[open_idx : close_idx + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict) and (obj.get("title") or obj.get("summary")):
        return obj
    return None


def _local_fallback(user_prompt: str, ai_response: str) -> dict:
    first_reply = re.sub(r"\s+", " ", (ai_response or "")).strip()
    title = (
        re.sub(r"\s+", " ", (user_prompt or "")).strip()[:70]
        or first_reply[:70]
        or "Saved conversation"
    )
    summary = first_reply[:280] or "No summary available."
    return {"title": title, "summary": summary}


async def summarize_turn(
    client: OllamaClient | None,
    user_prompt: str,
    ai_response: str,
) -> dict:
    """Return ``{"title": str, "summary": str}`` for a turn. Never raises."""
    fb = _local_fallback(user_prompt, ai_response)
    if client is None:
        return fb
    try:
        gen = await client.generate(
            _build_prompt(user_prompt, ai_response),
            json_mode=True,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception as exc:  # degrade gracefully — a card still gets a title
        slog.warning(
            event="summarizer_llm_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            degraded_mode=True,
        )
        return fb

    parsed = _parse(gen.answer)
    if not parsed:
        return fb
    return {
        "title": (str(parsed.get("title") or fb["title"]).strip())[:140],
        "summary": (str(parsed.get("summary") or fb["summary"]).strip())[:1000],
    }
