"""Claude tokenizer that uses Anthropic's ``count_tokens`` API endpoint.

Anthropic does not publish a fully local tokenizer for Claude 3+. The
official path is the API ``messages.count_tokens`` method. This wrapper
makes that call synchronously each time ``count()`` is invoked.

Cost / latency:
- The endpoint is free and rate-limited to 100 RPM (per Anthropic docs).
- Network roundtrip ~100-300ms per call. Use the ``cache`` arg to cache
  identical text→count mappings in process.

Fallback: when ``ANTHROPIC_API_KEY`` is not set, raises immediately so
the caller can fall back to a tiktoken approximation.
"""
import hashlib
import os

from orchestration.tokenizers.base import Tokenizer


DEFAULT_MODEL = "claude-opus-4-7"


class ClaudeAPITokenizerError(Exception):
    """Raised when the Anthropic count_tokens call fails."""


class ClaudeAPITokenizer(Tokenizer):
    """Claude tokenizer via Anthropic's ``count_tokens`` API.

    The Anthropic SDK is sync-only for count_tokens; we call it inline.
    For high-volume budgeting use the cached path or fall back to the
    tiktoken approximation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cache: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key is None:
            raise ClaudeAPITokenizerError(
                "ClaudeAPITokenizer requires ANTHROPIC_API_KEY"
            )
        self.model = model
        self._cache: dict[str, int] | None = {} if cache else None
        # Import-on-first-use so test environments without the SDK still load
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.api_key)

    @property
    def version(self) -> str:
        return f"anthropic-api:{self.model}"

    def count(self, text: str) -> int:
        if self._cache is not None:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in self._cache:
                return self._cache[key]

        try:
            resp = self._client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            n = int(resp.input_tokens)
        except Exception as exc:
            raise ClaudeAPITokenizerError(
                f"Anthropic count_tokens failed: {type(exc).__name__}: {exc}"
            ) from exc

        if self._cache is not None:
            self._cache[key] = n  # type: ignore[possibly-unbound]
        return n
