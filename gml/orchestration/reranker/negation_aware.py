"""NegationAwareReranker — adjust scores when question polarity disagrees
with memory polarity.

Why it helps
------------
AAL tuples carry a ``negated`` flag (e.g. "we DON'T use Redis" produces
``{subject: payments, verb: use, object: Redis, negated: True}``). The
cross-encoder doesn't read this flag, so when a user asks "do we use
Redis?", a memory saying "we DON'T use Redis" can score nearly identical
to "we use Redis" — both mention Redis, both surface "use", embedding
matches strongly. Adding logic that *demotes* polarity-mismatched memories
catches this class of failure.

Heuristic (no LLM)
------------------
- Detect query polarity via simple negation patterns (no, not, never,
  don't, doesn't, didn't).
- For each candidate, check ``raw_metadata.tuple.negated`` (AAL tuples
  carry this) OR scan content for the same negation patterns (raw memories).
- If polarities DISAGREE → multiply final_score by ``demotion_factor``
  (default 0.7).
- If polarities AGREE → small boost (×1.05) so genuine "X does NOT happen"
  evidence wins against irrelevant positive memories.

This is a thin wrapper around any base Reranker — composes with cross-encoder.
"""
import re

from orchestration.pipeline.contracts import (
    MemoryItem,
    Query,
    RankedHit,
    RetrievalHit,
)
from orchestration.reranker.base import Reranker


_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|don'?t|doesn'?t|didn'?t|hasn'?t|"
    r"haven'?t|isn'?t|aren'?t|wasn'?t|weren'?t|won'?t|wouldn'?t|"
    r"cannot|can'?t|couldn'?t|shouldn'?t)\b",
    re.IGNORECASE,
)


def is_negated_query(text: str) -> bool:
    """Cheap polarity check on the question text."""
    if not text:
        return False
    return bool(_NEGATION_RE.search(text))


def memory_is_negated(record: MemoryItem) -> bool:
    """Best-effort polarity check on a memory.

    1. If the memory is an AAL tuple, trust the structured ``negated`` flag.
    2. Otherwise look at the content for negation tokens.
    """
    rm = record.raw_metadata or {}
    tup = rm.get("tuple")
    if isinstance(tup, dict) and "negated" in tup:
        return bool(tup["negated"])
    return bool(_NEGATION_RE.search(record.content or ""))


class NegationAwareReranker(Reranker):
    """Adjusts a base reranker's output by query/memory polarity match.

    Composes around any reranker — call its ``pick_best`` first, then
    apply polarity adjustments. Demotion factor is conservative because
    negation detection is noisy ("not bad" is positive, "never used" is
    truly negative).
    """

    def __init__(
        self,
        base: Reranker,
        demotion_factor: float = 0.7,
        agreement_boost: float = 1.05,
    ) -> None:
        self.base = base
        self.demotion_factor = demotion_factor
        self.agreement_boost = agreement_boost

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        ranked = await self.base.pick_best(hits, query, k=k)
        if not ranked:
            return ranked

        q_neg = is_negated_query(query.text)
        adjusted: list[RankedHit] = []
        for rh in ranked:
            mem_neg = memory_is_negated(rh.hit.record)
            if q_neg == mem_neg:
                new_score = min(1.0, rh.final_score * self.agreement_boost)
                note = " polarity_agree"
            else:
                new_score = rh.final_score * self.demotion_factor
                note = " polarity_demote"
            adjusted.append(rh.model_copy(update={
                "final_score": new_score,
                "score_reason": rh.score_reason + note,
            }))
        adjusted.sort(key=lambda r: r.final_score, reverse=True)
        return adjusted
