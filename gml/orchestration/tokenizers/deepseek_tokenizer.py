"""Approximate Tokenizer for DeepSeek R1 models, backed by tiktoken cl100k_base.

DeepSeek R1 uses a BPE tokenizer derived from Qwen — not freely available
via a sync Python lib, so cl100k_base is used as a budgeting approximation.
The configured safety margin absorbs the divergence.
"""
import tiktoken

from orchestration.tokenizers.base import Tokenizer


class DeepSeekTokenizer(Tokenizer):
    _APPROX_TAG = "deepseek-approx"

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def version(self) -> str:
        return f"tiktoken:cl100k_base:{self._APPROX_TAG}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
