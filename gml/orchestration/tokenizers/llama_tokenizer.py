"""Approximate Tokenizer for Llama models, backed by tiktoken cl100k_base.

Llama uses a SentencePiece tokenizer; counts differ from cl100k_base by a
few percent for English text. We use cl100k_base as an approximation to
avoid the sentencepiece dependency + model-file shipping requirement; the
configured safety margin absorbs the divergence.
"""
import tiktoken

from orchestration.tokenizers.base import Tokenizer


class LlamaTokenizer(Tokenizer):
    _APPROX_TAG = "llama-approx"

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def version(self) -> str:
        return f"tiktoken:cl100k_base:{self._APPROX_TAG}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
