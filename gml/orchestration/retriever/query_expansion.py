"""Query expansion for recall.

Paraphrase the user's query with a cheap LLM into a few alternative phrasings,
retrieve for each, then merge the result sets with Reciprocal Rank Fusion (RRF)
so a memory that several phrasings agree on rises to the top.

GATED behind ``GML_QUERY_EXPANSION`` (default OFF): it adds one LLM round trip
plus N extra retrievals to the recall hot path and needs ``ANTHROPIC_API_KEY``.
Everything here is best-effort — any failure falls back to the original query.
"""
from __future__ import annotations

import os

from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import RetrievalHit

slog = StructuredLogger("retriever.query_expansion")

# "claude-haiku-3-5" — the cheap, fast paraphraser. Overridable.
_PARAPHRASE_MODEL = os.environ.get(
    "GML_QUERY_EXPANSION_MODEL", "claude-3-5-haiku-20241022"
)
DEFAULT_RRF_K = 60


def query_expansion_enabled() -> bool:
    return os.environ.get("GML_QUERY_EXPANSION", "0").lower() in {
        "1", "true", "yes", "on",
    }


async def expand_query(text: str, n: int = 3) -> list[str]:
    """Return up to ``n`` paraphrases of ``text`` (excluding the original).

    Best-effort: returns ``[]`` if the API key is missing or anything fails,
    so the caller transparently degrades to a single-query retrieval.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=api_key)
        prompt = (
            f"Rewrite this search query into {n} alternative phrasings someone "
            "might use to ask the same thing. Vary the wording and synonyms but "
            "keep the meaning identical. Return ONLY the rewrites, one per line, "
            f"with no numbering or commentary.\n\nQuery: {text}"
        )
        resp = await client.messages.create(
            model=_PARAPHRASE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        seen: set[str] = {text.strip().lower()}
        out: list[str] = []
        for line in raw.splitlines():
            cleaned = line.strip().lstrip("-•*0123456789. \t").strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                out.append(cleaned)
        return out[:n]
    except Exception as exc:  # never break recall on a paraphrase failure
        slog.warning(
            event="query_expansion_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return []


def rrf_merge(
    result_sets: list[list[RetrievalHit]], *, k: int, rrf_k: int = DEFAULT_RRF_K
) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion across several best-first ``RetrievalHit`` lists.

    Each contributing list votes ``1 / (rrf_k + rank)`` for its hits; a record
    surfaced by several paraphrases accumulates votes. Returns up to ``k`` hits
    ordered by fused score, with ``similarity`` overwritten by the normalized
    fused score (so downstream stages treat it uniformly, as HybridRetriever
    already does for its own fusion).
    """
    scores: dict[str, float] = {}
    rep: dict[str, RetrievalHit] = {}
    for hits in result_sets:
        for rank, hit in enumerate(hits):
            rid = hit.record.id
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
            rep.setdefault(rid, hit)
    if not scores:
        return []
    top = max(scores.values())
    fused = [
        rep[rid].model_copy(update={"similarity": (s / top) if top > 0 else 0.0})
        for rid, s in scores.items()
    ]
    fused.sort(key=lambda h: h.similarity, reverse=True)
    return fused[:k]
