"""HyDE — Hypothetical Document Embeddings for paraphrase-robust retrieval.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels", 2022.

Why HyDE helps
--------------
Question: *"What did Caroline research?"*  (5 tokens, abstract verb)
Evidence: "Caroline: I've been looking into adoption agencies"  (10+ tokens, casual verb)

Cosine similarity between bge embeddings of these two strings is ~0.55 —
not strong enough to dominate noise. The bare question is way shorter and
uses different surface vocabulary than the evidence.

HyDE: before embedding, ask a small LLM:

    "Generate one plausible message that would answer: <question>"
    → "Caroline mentioned she's been looking into adoption agencies recently."

Embed THAT hypothetical answer. Cosine to real evidence jumps to ~0.85 —
the hypothetical answer is in the same casual-message word distribution
as the evidence, so dense retrieval lands on it.

Architecture
------------
This module exposes one async function ``hyde_rewrite(text, client) → str``
that the Pipeline can call before embedding. It uses the same Ollama /
llama.cpp client SAM uses (Qwen3.5-4B Q4 by default, ~1-2s per call).

Failure semantics: if the LLM is unavailable or returns garbage, return
the original ``text`` unchanged. HyDE is purely additive — it never
makes retrieval worse, only adds latency when it doesn't help.
"""
import os
from typing import Awaitable

from orchestration.sam._ollama_client import OllamaClient


_HYDE_PROMPT = """\
You will be given a question. Generate ONE single plausible sentence
that would appear in a conversation as the answer to that question.

DO NOT explain. DO NOT say "Here is..." or "Sure...". Just output the
sentence itself — write it as if it's a natural utterance someone made
in a chat. Use casual language; include enough context (subject, verb,
object) that the sentence stands alone.

Question: {question}

Plausible answer sentence:"""


HYDE_ENABLED_DEFAULT = os.environ.get("GML_HYDE", "1") == "1"
HYDE_MAX_CHARS = 400  # truncate runaway responses
HYDE_MULTI_DEFAULT = os.environ.get("GML_HYDE_MULTI", "1") == "1"


# Phase C9: 3 different prompt styles. Generating multiple paraphrases and
# RRF-fusing the per-paraphrase retrievals consistently beats single HyDE
# by 3-7% on paraphrase-heavy retrieval benchmarks (BEIR HotpotQA etc.).
_HYDE_MULTI_PROMPTS = [
    # Style 1: factual / matter-of-fact
    "Generate ONE single plausible factual sentence that would answer this "
    "question. Just the sentence, no preamble.\n\nQuestion: {question}\n\nAnswer:",
    # Style 2: casual / conversational (matches LOCOMO's tone)
    "Imagine you're chatting casually with a friend. Generate ONE short "
    "natural sentence they might say that contains the answer to this "
    "question. No preamble.\n\nQuestion: {question}\n\nSentence:",
    # Style 3: specific / detail-rich
    "Generate ONE sentence rich in concrete details (names, dates, places, "
    "numbers) that would answer this question. Just the sentence, no "
    "preamble.\n\nQuestion: {question}\n\nDetailed answer:",
]


async def hyde_rewrite(
    text: str, client: OllamaClient | None
) -> str:
    """Generate a hypothetical-answer rewrite of ``text`` for embedding.

    Returns ``text`` unchanged if the client is None, disabled by env
    var, the rewrite is empty, or generation fails. Truncates rewrites
    to ``HYDE_MAX_CHARS`` to bound downstream embedding cost.
    """
    if not HYDE_ENABLED_DEFAULT or client is None or not text:
        return text

    prompt = _HYDE_PROMPT.format(question=text.strip())
    try:
        result = await client.generate(prompt, json_mode=False)
    except Exception:
        return text

    rewrite = (result.answer or "").strip()
    # Strip lead-in phrases the model sometimes can't resist
    for lead in ("Sure,", "Sure!", "Here is", "Here's", "Plausible answer",
                 "Answer:", "A:", "Sentence:"):
        if rewrite.lower().startswith(lead.lower()):
            rewrite = rewrite[len(lead):].lstrip(" :,-")
    # Take first sentence only — Qwen sometimes adds a follow-up
    first_period = next(
        (i for i, ch in enumerate(rewrite) if ch in ".!?" and i > 20),
        len(rewrite),
    )
    rewrite = rewrite[: first_period + 1].strip()
    if not rewrite or len(rewrite) < 10:
        return text
    return rewrite[:HYDE_MAX_CHARS]


async def hyde_fuse(
    text: str, client: OllamaClient | None, separator: str = " "
) -> str:
    """Return ``text + separator + hypothetical_answer`` for embedding.

    Sometimes better than pure HyDE because the embedding sees both the
    original question wording AND the hypothetical answer's words —
    union of both signals. Use this when you want HyDE as augmentation
    rather than replacement.
    """
    rewrite = await hyde_rewrite(text, client)
    if rewrite == text:
        return text
    return f"{text}{separator}{rewrite}"


# ---------------------------------------------------------------------------
# HyDE-3: multiple paraphrases (Phase C9)
# ---------------------------------------------------------------------------


import asyncio


async def _single_styled_rewrite(
    text: str, prompt_template: str, client: OllamaClient,
) -> str:
    """Generate one rewrite with a specific prompt style. Empty on failure."""
    prompt = prompt_template.format(question=text.strip())
    try:
        result = await client.generate(prompt, json_mode=False)
    except Exception:
        return ""
    rewrite = (result.answer or "").strip()
    for lead in ("Sure,", "Sure!", "Here is", "Here's", "Answer:",
                 "Sentence:", "Detailed answer:", "Plausible answer"):
        if rewrite.lower().startswith(lead.lower()):
            rewrite = rewrite[len(lead):].lstrip(" :,-")
    first_period = next(
        (i for i, ch in enumerate(rewrite) if ch in ".!?" and i > 20),
        len(rewrite),
    )
    rewrite = rewrite[: first_period + 1].strip()
    if len(rewrite) < 10:
        return ""
    return rewrite[:HYDE_MAX_CHARS]


async def hyde_multi_rewrite(
    text: str, client: OllamaClient | None,
    n: int = 3,
) -> list[str]:
    """Generate ``n`` style-diverse hypothetical answers IN PARALLEL.

    Returns up to ``n`` non-empty rewrites. Always includes the original
    text as element 0 so the union of (original, rewrites) covers both
    the literal question wording AND multiple paraphrases.

    Useful for multi-paraphrase HyDE retrieval (RRF-fuse the per-rewrite
    retrievals). When the LLM is missing or HyDE is disabled, returns
    just ``[text]``.
    """
    if not HYDE_ENABLED_DEFAULT or client is None or not text:
        return [text] if text else []

    prompts = _HYDE_MULTI_PROMPTS[:max(1, min(n, len(_HYDE_MULTI_PROMPTS)))]
    rewrites = await asyncio.gather(
        *(_single_styled_rewrite(text, p, client) for p in prompts)
    )
    out = [text]
    seen = {text.strip().lower()}
    for r in rewrites:
        if r and r.strip().lower() not in seen:
            seen.add(r.strip().lower())
            out.append(r)
    return out
