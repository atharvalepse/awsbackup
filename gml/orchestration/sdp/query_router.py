"""Query router — detect question type for category-aware retrieval.

LOCOMO measures recall by 5 categories. Each has a different retrieval
profile:

  cat 1 (single-hop)      → straightforward entity+fact lookup
  cat 2 (multi-hop)       → needs two facts joined; pull more candidates
  cat 3 (temporal)        → needs date/order info; bias toward recency
                            mentions in the candidate text
  cat 4 (open-domain)     → broad question; small top-k often enough
  cat 5 (adversarial)     → looks-misleading; treat like single-hop

This module classifies a question into a hint shape that downstream
retrieval / reranking can use to tune parameters. NO ML — pure regex
on question form. It's a heuristic that's right most of the time and
graceful when wrong.
"""
import re
from dataclasses import dataclass, field


_TEMPORAL_RE = re.compile(
    r"\b(when|what date|what day|what time|before|after|first|last|"
    r"recently|earliest|latest|next|previous|how long ago|since|until)\b",
    re.IGNORECASE,
)
_MULTI_HOP_RE = re.compile(
    r"\b(both|all|each|combine|together with|along with|"
    r"first.*then|after.*then|do they (?:also|both)|"
    r"\band\s+(?:also|then|next|the\s+\w+)\s+\b)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|never|no longer|don't|doesn't|didn't|hasn't|haven't|"
    r"isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"\b(how many|count|number of|total)\b", re.IGNORECASE,
)


@dataclass
class QueryHints:
    """Hints for downstream retrieval / reranking stages.

    Only ``top_k_multiplier`` is currently consumed by the Pipeline. Other
    booleans are inspected by the time-bucket retriever wrapper (Phase B7)
    and the answer-grounded scorer (Phase B8).
    """

    category: int = 1
    is_temporal: bool = False
    is_multi_hop: bool = False
    is_negation: bool = False
    is_count: bool = False

    # Retrieval-stage hints
    top_k_multiplier: float = 1.0

    notes: list[str] = field(default_factory=list)


def classify_query(text: str) -> QueryHints:
    """Classify a question into retrieval hints. Pure heuristic.

    Returns ``QueryHints`` with sensible defaults if no pattern matches.
    """
    hints = QueryHints()
    if not text:
        return hints
    t = text.strip()

    if _TEMPORAL_RE.search(t):
        hints.is_temporal = True
        hints.category = 3
        # For temporal questions: pull more candidates so dates aren't missed.
        # The time-bucket retriever wrapper (Phase B7) will use is_temporal
        # to bias toward time-near memories.
        hints.top_k_multiplier = 1.5
        hints.notes.append("temporal_question")

    if _MULTI_HOP_RE.search(t):
        hints.is_multi_hop = True
        hints.category = 2
        hints.top_k_multiplier = max(hints.top_k_multiplier, 1.5)
        hints.notes.append("multi_hop_signal")

    if _NEGATION_RE.search(t):
        hints.is_negation = True
        hints.notes.append("negation")

    if _COUNT_RE.search(t):
        hints.is_count = True
        hints.top_k_multiplier = max(hints.top_k_multiplier, 2.0)
        hints.notes.append("count_question_needs_full_recall")

    return hints
