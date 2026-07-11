"""Heuristic query decomposer for multi-hop questions.

Multi-hop LOCOMO questions often combine two or more facts:
  - "What yoga AND meditation does Caroline do?"     (conjunction)
  - "What are the two practices Caroline does?"       (cardinal hint)
  - "Where has Evan been on roadtrips and what cars has he owned?"  (compound)

A single retrieval pass biases toward whichever side of the conjunction
embeds best. Decomposing into 2-3 sub-questions, retrieving for each,
and merging gives the reranker a better candidate pool covering all the
hops. No LLM call — purely heuristic, ~50µs per query.

API:
    from orchestration.sdp.query_decomposer import decompose
    subqs = decompose("What yoga and meditation does Caroline do?")
    # → ["What yoga does Caroline do?",
    #    "What meditation does Caroline do?"]

When the question can't be cleanly split, returns the original wrapped
in a 1-element list so callers don't have to special-case it.

Intentionally conservative: emits at most 3 sub-queries, never
decomposes single-hop questions (the splits would just dilute retrieval).
"""
import re


# Conjunctions to split on. Match " AND ", " OR " between word-boundary tokens.
# Anchored on spaces so we don't split inside compound names ("Stripe and Adyen"
# — but actually we DO want to split that for retrieval; the cost is low).
_AND_RE = re.compile(r"\s+\b(?:and|or)\b\s+", re.IGNORECASE)

# Phrases that indicate "multi-item answer expected" — used to widen the
# decomposition heuristic to cover non-conjunction multi-hop questions.
_MULTI_ITEM_HINTS = (
    "what kinds", "what sorts", "what types",
    "what hobbies", "what activities", "what things",
    "what practices", "what items", "what places",
    "what cars", "what sports",
    "two ", "three ", "four ", "five ", "six ", "all the ",
)


def _looks_multi_item(question: str) -> bool:
    """Heuristic: question expects a list of items in the answer."""
    q_lower = question.lower()
    return any(hint in q_lower for hint in _MULTI_ITEM_HINTS)


def _split_on_conjunction(text: str) -> list[str]:
    """Split on top-level AND/OR, keeping pieces non-empty."""
    parts = _AND_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def decompose(question: str, max_subqueries: int = 3) -> list[str]:
    """Return 1-N sub-questions. Always returns at least the original.

    Heuristic strategies tried in order:
      1. Top-level AND/OR split  (best for conjunction questions)
      2. Multi-item rewording    (asks for "other X" if the original
                                  implies a list, e.g. cardinal numbers)
      3. Fallback: return [question]

    Always includes the ORIGINAL question as the first element so the
    canonical phrasing dominates score-merging downstream.
    """
    q = (question or "").strip()
    if not q:
        return [q]
    q_no_qmark = q.rstrip("?").rstrip(".").strip()

    # ── Strategy 1: AND/OR split ──────────────────────────────────────
    # Only fires when both sides look like substantive content (>= 3
    # words) — guards against "yes and no" or "or what?" type junk.
    parts = _split_on_conjunction(q_no_qmark)
    if len(parts) >= 2 and all(len(p.split()) >= 2 for p in parts):
        # Re-add the trailing ? to each split for natural phrasing.
        subs = [(p + "?") for p in parts]
        # Always include the original (canonical) first
        result = [q] + [s for s in subs if s.lower() != q.lower()]
        return result[:max_subqueries]

    # ── Strategy 2: multi-item rewording ──────────────────────────────
    # Examples:
    #   "What two practices does Caroline do?" →
    #     ["What two practices does Caroline do?",
    #      "What practices does Caroline do?",            (drop the count)
    #      "What other practices does Caroline do?"]      (ask for more)
    if _looks_multi_item(q_no_qmark):
        rewrites: list[str] = []
        # Drop cardinal modifiers ("two practices" → "practices")
        no_count = re.sub(r"\b(two|three|four|five|six|seven)\s+",
                          "", q_no_qmark, flags=re.IGNORECASE)
        if no_count != q_no_qmark and len(no_count.split()) >= 3:
            rewrites.append(no_count + "?")
        # Insert "all" if not present, to broaden ("What practices..." → "What ALL practices...")
        if " all " not in q.lower() and "what " in q_no_qmark.lower():
            broadened = re.sub(
                r"\b(what)\b", "what all", q_no_qmark, count=1, flags=re.IGNORECASE
            )
            if broadened != q_no_qmark:
                rewrites.append(broadened + "?")
        if rewrites:
            result = [q] + [r for r in rewrites if r.lower() != q.lower()]
            return result[:max_subqueries]

    # ── Fallback ──────────────────────────────────────────────────────
    return [q]
