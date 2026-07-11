"""Approximate Tokenizer for Gemini models, backed by tiktoken cl100k_base.

Gemini does not expose a freely-available synchronous tokenizer, so this is
an APPROXIMATION. Real Gemini tokenization can diverge from cl100k_base; the
configured safety margin absorbs the difference.
"""
import tiktoken

from orchestration.tokenizers.base import Tokenizer


class GeminiTokenizer(Tokenizer):
    _APPROX_TAG = "gemini-approx"

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def version(self) -> str:
        return f"tiktoken:cl100k_base:{self._APPROX_TAG}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
