"""Minimal async HTTP clients for SAM's local-LLM reasoner.

Two backends are supported:

- ``HTTPOllamaClient`` — talks to Ollama at ``/api/generate`` (port 11434
  by default). The original integration; kept for backwards compat.
- ``LlamaCppClient`` — talks to llama.cpp's ``llama-server`` via its
  OpenAI-compatible ``/v1/chat/completions`` endpoint (port 8080 by
  default). The new default.

Choose the backend at runtime with env var ``GML_LLM_BACKEND``:
  ``llamacpp`` (default) or ``ollama``.

Both return a :class:`GenerationResult` so callers don't need to know
which backend ran the prompt. The ``<think>...</think>`` splitter still
runs for both — Qwen3 family models emit a thinking block.

The class names ``OllamaClient`` / ``HTTPOllamaClient`` are kept for
backwards compatibility with the rest of the codebase, which historically
used "Ollama" as a synonym for "local LLM."
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


# ---------------------------------------------------------------------------
# Per-backend defaults. Override individually with env vars.
# ---------------------------------------------------------------------------

# Backend default switched to Ollama after we measured:
#  - qwen2.5:3b on Ollama: 241ms per answer-gen call
#  - qwen3:4b on Ollama:   6074ms per answer-gen call
#  - qwen3.5-4b-q4 on llama.cpp: ~7s per call AND llama.cpp serializes
#    so concurrent requests time out
# Ollama with qwen2.5:3b is 25x faster than the llama.cpp alternative and
# produces correct answers on our smoke tests.
DEFAULT_BACKEND = os.environ.get("GML_LLM_BACKEND", "ollama").lower()

# llama.cpp server (port 8080 is the llama-server default) — kept as a
# backend option, but no longer the default.
LLAMACPP_BASE_URL = os.environ.get("GML_LLAMACPP_BASE_URL", "http://127.0.0.1:8080")
LLAMACPP_MODEL = os.environ.get("GML_LLAMACPP_MODEL", "qwen3.5-4b-q4")

# Ollama (port 11434 is the Ollama default) — new default backend.
OLLAMA_BASE_URL = os.environ.get("GML_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("GML_OLLAMA_MODEL", "qwen2.5:3b")

# Back-compat aliases — modules still importing these continue to work.
DEFAULT_BASE_URL = LLAMACPP_BASE_URL if DEFAULT_BACKEND == "llamacpp" else OLLAMA_BASE_URL
DEFAULT_MODEL = (
    os.environ.get("GML_SAM_MODEL")  # legacy override
    or (LLAMACPP_MODEL if DEFAULT_BACKEND == "llamacpp" else OLLAMA_MODEL)
)
DEFAULT_TIMEOUT_SECONDS = 90.0
# This system extracts/reasons over FACTS — Ollama's 0.8 default temperature
# causes hallucination and unstable JSON (which fails parsing → 0 memories).
# Default low; callers that want determinism pass temperature=0.0 explicitly.
DEFAULT_TEMPERATURE = float(os.environ.get("GML_LLM_TEMPERATURE", "0.1"))

# Qwen3 / Qwen3.5 family ship "thinking mode" on by default, generating
# hundreds of <think> tokens before the answer. SAM doesn't need that —
# it makes calls 5-10x slower. Disable by default; set GML_ENABLE_THINKING=1
# to turn it back on.
ENABLE_THINKING_DEFAULT = os.environ.get("GML_ENABLE_THINKING", "0") == "1"


@dataclass(frozen=True)
class GenerationResult:
    """Decomposed Ollama response. ``thinking`` is the R1 ``<think>`` block
    (empty for non-R1 models); ``answer`` is the user-visible response text."""

    thinking: str
    answer: str

    @property
    def raw(self) -> str:
        if not self.thinking:
            return self.answer
        return f"<think>{self.thinking}</think>{self.answer}"


class OllamaClient(ABC):
    """Strategy interface — real and mock clients implement this."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult: ...


def _split_thinking(raw: str) -> GenerationResult:
    """Parse a ``<think>...</think>`` prefix out of an R1 response."""
    open_tag = "<think>"
    close_tag = "</think>"
    if open_tag in raw and close_tag in raw:
        start = raw.index(open_tag) + len(open_tag)
        end = raw.index(close_tag)
        thinking = raw[start:end].strip()
        answer = raw[end + len(close_tag):].strip()
        return GenerationResult(thinking=thinking, answer=answer)
    return GenerationResult(thinking="", answer=raw.strip())


class HTTPOllamaClient(OllamaClient):
    """Ollama backend. Uses ``/api/generate`` with stream=false."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            body["format"] = "json"
        # Ollama exposes generation caps under `options.num_predict`. Without
        # this Qwen2.5:3b happily produces 200+ tokens for what should be a
        # one-word answer, which destroys token-F1 precision against short
        # LOCOMO golds.
        options: dict = {}
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        # Always pin temperature (low by default) — leaving it unset lets Ollama
        # use 0.8, which drives hallucination + flaky JSON.
        options["temperature"] = float(
            temperature if temperature is not None else DEFAULT_TEMPERATURE
        )
        if seed is not None:
            options["seed"] = int(seed)
        if options:
            body["options"] = options

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=body)
            resp.raise_for_status()
            data = resp.json()

        raw = data.get("response", "")
        return _split_thinking(raw)


class LlamaCppClient(OllamaClient):
    """llama.cpp ``llama-server`` backend.

    Talks to the OpenAI-compatible ``/v1/chat/completions`` endpoint so
    we get a stable interface (llama.cpp's native ``/completion`` API
    has shifted historically). Model name doesn't actually matter to
    llama-server — it serves whatever .gguf was loaded at startup — but
    we send one anyway so logs show what we expect to be hitting.
    """

    def __init__(
        self,
        base_url: str = LLAMACPP_BASE_URL,
        model: str = LLAMACPP_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # SAM/extractor outputs are ~150-300 tokens; cap aggressively so
            # a runaway generation can't blow latency. Callers can override
            # per-request via ``max_tokens`` (e.g. answer-gen wants ~20).
            "max_tokens": int(max_tokens) if max_tokens is not None else 512,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if temperature is not None:
            body["temperature"] = float(temperature)
        if seed is not None:
            body["seed"] = int(seed)

        # Qwen3 thinking-mode toggle — llama.cpp forwards chat_template_kwargs
        # straight to the GGUF's jinja template, which honors enable_thinking.
        if not ENABLE_THINKING_DEFAULT:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions", json=body
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"llama-server returned unexpected shape: {data!r}"
            ) from exc

        # Qwen3 family puts reasoning in a separate `reasoning_content` field
        # on the OpenAI-compatible endpoint, not inline as `<think>` tags.
        thinking = (message.get("reasoning_content") or "").strip()
        answer = (message.get("content") or "").strip()

        # Some configs still emit inline <think>; handle that too.
        if not thinking and "<think>" in answer:
            return _split_thinking(answer)
        return GenerationResult(thinking=thinking, answer=answer)


# ---------------------------------------------------------------------------
# Factory — pick the backend by env var so every consumer goes through one
# call site. Used by SAM, MemoryExtractor, and the analyze() MCP tool.
# ---------------------------------------------------------------------------


def make_local_llm_client(
    backend: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> "OllamaClient":
    """Build the right client based on backend (env var by default)."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "llamacpp":
        return LlamaCppClient(
            base_url=base_url or LLAMACPP_BASE_URL,
            model=model or LLAMACPP_MODEL,
            timeout_seconds=timeout_seconds,
        )
    if backend == "ollama":
        return HTTPOllamaClient(
            base_url=base_url or OLLAMA_BASE_URL,
            model=model or OLLAMA_MODEL,
            timeout_seconds=timeout_seconds,
        )
    if backend == "transformers":
        # In-process transformers-served LLM. Used to load the LoRA-tuned
        # Qwen2.5-3B from FT-2 without needing to convert to GGUF/Ollama.
        # See orchestration/sam/transformers_client.py.
        from orchestration.sam.transformers_client import (
            TransformersLLMClient,
        )
        return TransformersLLMClient(
            base_model=os.environ.get("GML_TRANSFORMERS_BASE", "Qwen/Qwen2.5-3B-Instruct"),
            adapter_path=os.environ.get("GML_TRANSFORMERS_ADAPTER") or None,
            device=os.environ.get("GML_TRANSFORMERS_DEVICE", "mps"),
        )
    raise ValueError(
        f"Unknown GML_LLM_BACKEND={backend!r} (expected 'llamacpp', 'ollama', or 'transformers')"
    )


def make_answer_llm_client(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple["OllamaClient", bool]:
    """Build a *separate* LLM client for answer-gen (decoupled from ingest LLM).

    Lets the bench use FT'd Qwen for cheap ingest (SAM summaries, HyDE,
    entity-synth) while sending the final answer-gen prompt to a stronger
    model (e.g. gemma2:27b via Ollama). Falls back to the ingest LLM when
    no override is set.

    Env vars (all optional):
      - GML_ANSWER_LLM_BACKEND  : "ollama" | "transformers" | "llamacpp" (default: same as ingest)
      - GML_ANSWER_OLLAMA_MODEL : ollama tag for the answer model
      - GML_ANSWER_OLLAMA_URL   : override ollama base URL
      - GML_ANSWER_LLM_USES_FT_PROMPT : "1" forces FT training prompt
                                        (auto-detected if model id contains "locomo-ft")

    Returns: (client, uses_ft_prompt) — second element tells the
    AnswerGenerator which prompt shape to use.
    """
    override_backend = os.environ.get("GML_ANSWER_LLM_BACKEND", "").strip().lower()
    if not override_backend:
        # No separate answer-gen client — reuse the ingest one
        client = make_local_llm_client(timeout_seconds=timeout_seconds)
        # FT prompt selection mirrors AnswerGenerator's legacy logic
        _ingest_backend = (os.environ.get("GML_LLM_BACKEND", "") or DEFAULT_BACKEND).lower()
        _ollama_model = os.environ.get("GML_OLLAMA_MODEL", OLLAMA_MODEL).lower()
        uses_ft = (
            _ingest_backend == "transformers"
            or os.environ.get("GML_LLM_USES_FT_PROMPT", "0") == "1"
            or (_ingest_backend == "ollama" and "locomo-ft" in _ollama_model)
        )
        return client, uses_ft

    # Separate answer-gen client
    if override_backend == "ollama":
        model = os.environ.get("GML_ANSWER_OLLAMA_MODEL", OLLAMA_MODEL)
        base_url = os.environ.get("GML_ANSWER_OLLAMA_URL", OLLAMA_BASE_URL)
        client = HTTPOllamaClient(
            base_url=base_url, model=model, timeout_seconds=timeout_seconds,
        )
        uses_ft = (
            os.environ.get("GML_ANSWER_LLM_USES_FT_PROMPT", "0") == "1"
            or "locomo-ft" in model.lower()
        )
        return client, uses_ft

    if override_backend == "transformers":
        from orchestration.sam.transformers_client import TransformersLLMClient
        client = TransformersLLMClient(
            base_model=os.environ.get("GML_ANSWER_TRANSFORMERS_BASE",
                os.environ.get("GML_TRANSFORMERS_BASE", "Qwen/Qwen2.5-3B-Instruct")),
            adapter_path=os.environ.get("GML_ANSWER_TRANSFORMERS_ADAPTER")
                or os.environ.get("GML_TRANSFORMERS_ADAPTER") or None,
            device=os.environ.get("GML_ANSWER_TRANSFORMERS_DEVICE",
                os.environ.get("GML_TRANSFORMERS_DEVICE", "mps")),
        )
        return client, True  # transformers backend almost always = our FT'd LoRA

    if override_backend == "llamacpp":
        client = LlamaCppClient(
            base_url=os.environ.get("GML_ANSWER_LLAMACPP_URL", LLAMACPP_BASE_URL),
            model=os.environ.get("GML_ANSWER_LLAMACPP_MODEL", LLAMACPP_MODEL),
            timeout_seconds=timeout_seconds,
        )
        uses_ft = os.environ.get("GML_ANSWER_LLM_USES_FT_PROMPT", "0") == "1"
        return client, uses_ft

    raise ValueError(
        f"Unknown GML_ANSWER_LLM_BACKEND={override_backend!r}"
    )


def health_probe_url(backend: str | None = None) -> str:
    """Return the URL to probe for the configured backend's liveness."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "llamacpp":
        return f"{LLAMACPP_BASE_URL}/health"
    return f"{OLLAMA_BASE_URL}/api/tags"


class MockOllamaClient(OllamaClient):
    """Test client that returns canned responses. Records every prompt sent."""

    def __init__(self, responses: list[GenerationResult] | None = None) -> None:
        self.responses: list[GenerationResult] = list(responses) if responses else []
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("MockOllamaClient: no canned responses left")
        return self.responses.pop(0)

    def queue(self, *, thinking: str = "", answer: str | dict = "") -> None:
        """Append a canned response. If ``answer`` is a dict, it is JSON-encoded."""
        if isinstance(answer, dict):
            answer = json.dumps(answer)
        self.responses.append(GenerationResult(thinking=thinking, answer=answer))
