"""Regex-based Classifier — used standalone and as the LLM fallback."""
import re

from orchestration.classifier.base import Classifier
from orchestration.pipeline.contracts import Classification, ClassificationSource, Query


_KEYWORD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(fix|debug|error|bug|crash|broken|fail|stack ?trace)\b", re.IGNORECASE), "debugging"),
    (re.compile(r"\b(write|draft|compose|email|essay|letter)\b", re.IGNORECASE), "writing"),
    (re.compile(r"\b(code|implement|build|refactor|function|class|method)\b", re.IGNORECASE), "coding"),
    (re.compile(r"\b(research|investigate|find\b|search|look\s?up|sources?)\b", re.IGNORECASE), "research"),
    (re.compile(r"\b(do|finish|complete|task|todo|action)\b", re.IGNORECASE), "task"),
    (re.compile(r"\b(what|why|how|when|where|who|explain|describe)\b", re.IGNORECASE), "question"),
]


def _match_keyword(text: str, *, degraded: bool = False) -> Classification:
    for pattern, intent_type in _KEYWORD_PATTERNS:
        if pattern.search(text):
            return Classification(
                intent_type=intent_type,
                entities=[],
                retrieval_hints={},
                confidence=0.3,
                source=ClassificationSource.KEYWORD_FALLBACK,
                degraded=degraded,
            )
    return Classification(
        intent_type="other",
        entities=[],
        retrieval_hints={},
        confidence=0.3,
        source=ClassificationSource.KEYWORD_FALLBACK,
        degraded=degraded,
    )


class KeywordClassifier(Classifier):
    """Pure-regex Classifier. Always succeeds, never calls out to an LLM."""

    async def classify(self, query: Query) -> Classification:
        return _match_keyword(query.text)
