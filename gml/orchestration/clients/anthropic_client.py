"""Claude target-AI client via the Anthropic SDK."""
import os
import time

from orchestration.clients.base import AssistantResponse, Client
from orchestration.errors import OrchestrationError
from orchestration.pipeline.contracts import TranslatedPayload


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096


class AnthropicClientError(OrchestrationError):
    """Raised when the Anthropic SDK call fails."""


class AnthropicClient(Client):
    """Claude via Anthropic. Reads ``ANTHROPIC_API_KEY`` from env by default."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens

    async def send(self, payload: TranslatedPayload) -> AssistantResponse:
        if self.api_key is None:
            raise AnthropicClientError("ANTHROPIC_API_KEY is not set")

        from anthropic import AsyncAnthropic

        model = self.model or payload.target.model_version or DEFAULT_MODEL
        client = AsyncAnthropic(api_key=self.api_key)

        # Pass the formatted context as a system message; the rewritten user
        # query goes as the user message.
        t0 = time.perf_counter()
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=payload.formatted_context,
                messages=[{"role": "user", "content": payload.user_query}],
            )
        except Exception as exc:
            raise AnthropicClientError(
                f"Anthropic call failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        return AssistantResponse(
            text=text,
            target=payload.target,
            model_version=model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            raw_metadata={"stop_reason": getattr(resp, "stop_reason", None)},
        )
