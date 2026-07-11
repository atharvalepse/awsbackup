"""SDP Stages 7 + 8 — ImportanceScorer + ConfidenceScorer.

Both produce a float in [0, 1] from a SemanticUnit-shaped dict.

ImportanceScorer asks "how much should orchestration prioritize this?".
ConfidenceScorer asks "how sure are we that we extracted this correctly?".

Both are heuristic, deterministic, no LLM.
"""


_HIGH_VALUE_CATEGORIES = {"port", "url", "person", "version"}
_PROJECT_KEYWORDS = {
    "production", "prod", "staging", "deploy", "deploys", "deployed",
    "auth", "billing", "payments", "orders", "gateway", "scheduler",
}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class ImportanceScorer:
    """Score how important a memory is for future orchestration.

    Higher score = more likely to survive trimming + win reranking.
    """

    def score(self, unit: dict) -> float:
        score = 0.5  # neutral baseline

        # Configuration / infrastructure facts matter more than throwaway facts
        if unit.get("category") in _HIGH_VALUE_CATEGORIES:
            score += 0.2

        # Decisions and issues are sticky — they tend to be referenced later
        unit_type = unit.get("type")
        if unit_type == "decision":
            score += 0.2
        elif unit_type == "issue":
            score += 0.15
        elif unit_type == "preference":
            score += 0.1

        # Project-keyword signal in the surrounding text
        span = (unit.get("span") or "").lower()
        if any(kw in span for kw in _PROJECT_KEYWORDS):
            score += 0.1

        return _clamp(score)


class ConfidenceScorer:
    """Score how reliable an extraction is.

    Higher = the heuristic is confident the extraction matches the source.
    Used downstream for conflict resolution and provenance.
    """

    def score(self, unit: dict) -> float:
        unit_type = unit.get("type")

        # Direct factual extractions (versions, ports, URLs) are deterministic
        # — regex either matched or it didn't.
        if unit.get("category") in _HIGH_VALUE_CATEGORIES:
            base = 0.95
        elif unit_type == "fact":
            base = 0.85
        elif unit_type == "decision":
            base = 0.8
        elif unit_type == "issue":
            base = 0.75
        elif unit_type == "preference":
            base = 0.7
        else:
            base = 0.6

        # If we have both subject and attribute, we're more confident in the
        # extraction's structure.
        if unit.get("subject") and unit.get("attribute"):
            base += 0.05

        return _clamp(base)
