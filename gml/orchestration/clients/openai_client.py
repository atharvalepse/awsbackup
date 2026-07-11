"""GPT target-AI client via the OpenAI SDK."""
import os
import time

from orchestration.clients.base import AssistantResponse, Client
from orchestration.errors import OrchestrationError
from orchestration.pipeline.contracts import TranslatedPayload


DEFAULT_MODEL = "gpt-4o"


class OpenAIClientError(OrchestrationError):
    """Raised when the OpenAI SDK call fails."""


class OpenAIClient(Client):
    """GPT via OpenAI. Reads ``OPENAI_API_KEY`` from env by default."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    async def send(self, payload: TranslatedPayload) -> AssistantResponse:
        if self.api_key is None:
            raise OpenAIClientError("OPENAI_API_KEY is not set")

        from openai import AsyncOpenAI

        model = self.model or payload.target.model_version or DEFAULT_MODEL
        client = AsyncOpenAI(api_key=self.api_key)

        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": payload.formatted_context},
                    {"role": "user", "content": payload.user_query},
                ],
            )
        except Exception as exc:
            raise OpenAIClientError(
                f"OpenAI call failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return AssistantResponse(
            text=text,
            target=payload.target,
            model_version=model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            raw_metadata={"finish_reason": choice.finish_reason},
        )
