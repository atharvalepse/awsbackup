"""tiktoken-backed Tokenizer for GPT-family models."""
import logging

import tiktoken

from orchestration.tokenizers.base import Tokenizer


logger = logging.getLogger(__name__)

_O200K_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")
_CL100K_PREFIXES = ("gpt-3.5",)
_CL100K_EXACT = frozenset({"gpt-4"})


def _resolve_encoding_name(model_version: str) -> str:
    mv = model_version.lower()
    if mv.startswith(_O200K_PREFIXES):
        return "o200k_base"
    if mv in _CL100K_EXACT or mv.startswith(_CL100K_PREFIXES):
        return "cl100k_base"
    logger.warning(
        "TiktokenTokenizer: unknown model_version %r, defaulting to cl100k_base",
        model_version,
    )
    return "cl100k_base"


class TiktokenTokenizer(Tokenizer):
    """tiktoken Tokenizer; encoding selected from ``model_version``."""

    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        self._encoding_name = _resolve_encoding_name(model_version)
        self._encoding = tiktoken.get_encoding(self._encoding_name)

    @property
    def version(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
