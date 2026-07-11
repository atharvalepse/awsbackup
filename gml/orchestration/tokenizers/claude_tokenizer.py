"""Approximate Tokenizer for Claude models, backed by tiktoken cl100k_base.

Anthropic does not publish a free synchronous tokenizer; cl100k_base is a
reasonable approximation for budgeting English text. The configured safety
margin absorbs the divergence.
"""
import tiktoken

from orchestration.tokenizers.base import Tokenizer


class ClaudeTokenizer(Tokenizer):
    _APPROX_TAG = "claude-approx"

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def version(self) -> str:
        return f"tiktoken:cl100k_base:{self._APPROX_TAG}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
