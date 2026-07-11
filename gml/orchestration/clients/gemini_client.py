"""Gemini target-AI client via google-genai (same SDK as Classifier+Embedder)."""
import os
import time

from orchestration.clients.base import AssistantResponse, Client
from orchestration.errors import OrchestrationError
from orchestration.pipeline.contracts import TranslatedPayload


DEFAULT_MODEL = "gemini-2.5-pro"


class GeminiClientError(OrchestrationError):
    """Raised when the Gemini SDK call fails."""


class GeminiClient(Client):
    """Gemini via google-genai. Reads ``GEMINI_API_KEY`` from env by default."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model

    async def send(self, payload: TranslatedPayload) -> AssistantResponse:
        if self.api_key is None:
            raise GeminiClientError("GEMINI_API_KEY is not set")

        from google import genai
        from google.genai import types as genai_types

        model = self.model or payload.target.model_version or DEFAULT_MODEL
        client = genai.Client(api_key=self.api_key)

        # Gemini uses a single contents string with the system instructions
        # baked in via system_instruction.
        config = genai_types.GenerateContentConfig(
            system_instruction=payload.formatted_context,
        )

        t0 = time.perf_counter()
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=[payload.user_query],
                config=config,
            )
        except Exception as exc:
            raise GeminiClientError(
                f"Gemini call failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        return AssistantResponse(
            text=text,
            target=payload.target,
            model_version=model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            raw_metadata={},
        )
