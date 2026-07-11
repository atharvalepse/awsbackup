"""Default :class:`Client` resolution by target family.

Most callers want "give me the right client for this target" — they don't
care about per-family construction. This factory dispatches on
``target.model_family``; pass a custom mapping when you need to override.
"""
import os

from orchestration.clients.anthropic_client import AnthropicClient
from orchestration.clients.base import Client
from orchestration.clients.gemini_client import GeminiClient
from orchestration.clients.ollama_client import OllamaClient
from orchestration.clients.openai_client import OpenAIClient
from orchestration.pipeline.contracts import ModelFamily, TargetDescriptor


def build_default_client_for_target(target: TargetDescriptor) -> Client:
    """Return a fresh Client matched to ``target.model_family``.

    GPT / CURSOR → :class:`OpenAIClient`
    CLAUDE       → :class:`AnthropicClient`
    GEMINI       → :class:`GeminiClient`
    LLAMA, DEEPSEEK → :class:`OllamaClient`

    API keys are read from env by each client.
    """
    family = target.model_family
    if family == ModelFamily.GPT:
        return OpenAIClient()
    if family == ModelFamily.CLAUDE:
        return AnthropicClient()
    if family == ModelFamily.GEMINI:
        return GeminiClient()
    if family in (ModelFamily.LLAMA, ModelFamily.DEEPSEEK):
        # Cost-saving default: route local Ollama targets at the model actually
        # loaded in the daemon (GML_TARGET_OLLAMA_MODEL, else the ingest/answer
        # model GML_OLLAMA_MODEL) instead of the descriptor's hardcoded tag.
        local_model = (
            os.environ.get("GML_TARGET_OLLAMA_MODEL")
            or os.environ.get("GML_OLLAMA_MODEL")
            or None
        )
        return OllamaClient(model=local_model)
    if family == ModelFamily.CURSOR:
        # Cursor doesn't have a dedicated API; tunnel through the backend
        # the Cursor target was configured with (GPT-family today).
        return OpenAIClient(model=target.cursor_backend)
    raise NotImplementedError(f"No default client for model_family={family!r}")
