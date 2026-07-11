"""Ollama target-AI client — serves any model pulled into the local daemon.

Used for the DEEPSEEK and LLAMA target families. The Ollama daemon is the
same one SAM uses for its DeepSeek R1 reasoner; we keep separate client
classes for clarity since one ships a payload to the target AI and the
other does internal reasoning.

Reads target.model_version as the Ollama model name when ``model`` is not
explicitly set (e.g. ``"deepseek-r1:8b"``, ``"llama-3.3-70b"`` → user must
pull the matching model).
"""
import os
import time

import httpx

from orchestration.clients.base import AssistantResponse, Client
from orchestration.errors import OrchestrationError
from orchestration.pipeline.contracts import TranslatedPayload


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaClientError(OrchestrationError):
    """Raised when the Ollama daemon call fails."""


class OllamaClient(Client):
    """Local-Ollama target-AI client. Strips ``<think>`` blocks from R1 output."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def send(self, payload: TranslatedPayload) -> AssistantResponse:
        model = self.model or payload.target.model_version
        if not model:
            raise OllamaClientError(
                "OllamaClient requires a model — set client.model or "
                "target.model_version (e.g. 'deepseek-r1:8b')"
            )

        options = {
            "num_predict": int(os.environ.get("GML_CHAT_MAX_TOKENS", "512")),
            "temperature": float(os.environ.get("GML_CHAT_TEMPERATURE", "0.2")),
        }
        # When the target IS the LOCOMO-fine-tuned model, the generic
        # system/prompt split makes it ignore the injected memory and
        # hallucinate — it was trained on a single CONTEXT/QUESTION/ANSWER
        # prompt. Reproduce that format so it actually grounds on memory.
        use_ft = (
            os.environ.get("GML_LLM_USES_FT_PROMPT", "0") == "1"
            or "locomo-ft" in model.lower()
            or "qwen-ft" in model.lower()
        )
        if use_ft:
            ft_prompt = (
                "Below is some context from a long-running conversation, "
                "followed by a question. Answer the question concisely.\n\n"
                f"CONTEXT:\n{payload.formatted_context}\n\n"
                f"QUESTION: {payload.user_query}\n\nANSWER:"
            )
            body = {"model": model, "prompt": ft_prompt, "stream": False, "options": options}
        else:
            body = {
                "model": model,
                "system": payload.formatted_context,
                "prompt": payload.user_query,
                "stream": False,
                "options": options,
            }

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise OllamaClientError(
                f"Ollama call failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)

        raw = data.get("response", "")
        text = _strip_thinking(raw)
        return AssistantResponse(
            text=text,
            target=payload.target,
            model_version=model,
            latency_ms=latency_ms,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            raw_metadata={"done": data.get("done"), "raw_response_with_thinking": raw},
        )


def _strip_thinking(text: str) -> str:
    """Remove a leading ``<think>...</think>`` block. DeepSeek R1 emits one;
    most other Ollama models don't. Either way we return the user-visible
    answer."""
    open_tag = "<think>"
    close_tag = "</think>"
    if open_tag in text and close_tag in text:
        end = text.index(close_tag) + len(close_tag)
        return text[end:].strip()
    return text.strip()
