"""Translator — dispatch AssembledContext to the right per-target adapter.

This is the ONLY module that knows about target-specific formatting. The
Pipeline gives it a context; the Translator looks up the adapter for the
context's ``target.model_family`` and produces a TranslatedPayload.
"""
import orchestration
from orchestration.pipeline.contracts import (
    AssembledContext,
    ModelFamily,
    TargetDescriptor,
    TranslatedPayload,
)
from orchestration.translator.base import TranslatorAdapter
from orchestration.translator.claude_adapter import ClaudeAdapter
from orchestration.translator.deepseek_adapter import DeepSeekAdapter
from orchestration.translator.gemini_adapter import GeminiAdapter
from orchestration.translator.gpt_adapter import GPTAdapter
from orchestration.translator.llama_adapter import LlamaAdapter


def default_adapters() -> dict[ModelFamily, TranslatorAdapter]:
    """Built-in adapter set covering GPT, Gemini, Claude, Llama, DeepSeek.
    Cursor aliases to the GPT adapter."""
    gpt = GPTAdapter()
    return {
        ModelFamily.GPT: gpt,
        ModelFamily.GEMINI: GeminiAdapter(),
        ModelFamily.CLAUDE: ClaudeAdapter(),
        ModelFamily.LLAMA: LlamaAdapter(),
        ModelFamily.DEEPSEEK: DeepSeekAdapter(),
        ModelFamily.CURSOR: gpt,
    }


class Translator:
    """Strategy dispatcher. Pipeline owns one Translator; per-target adapter
    is resolved at translate-time from the context's TargetDescriptor."""

    def __init__(
        self, adapters: dict[ModelFamily, TranslatorAdapter] | None = None
    ) -> None:
        self.adapters: dict[ModelFamily, TranslatorAdapter] = (
            adapters if adapters is not None else default_adapters()
        )

    def adapter_for(self, target: TargetDescriptor) -> TranslatorAdapter:
        try:
            return self.adapters[target.model_family]
        except KeyError as exc:
            raise NotImplementedError(
                f"No Translator adapter registered for model_family={target.model_family!r}"
            ) from exc

    def translate(
        self, context: AssembledContext, config_hash: str
    ) -> TranslatedPayload:
        adapter = self.adapter_for(context.query.target)
        formatted = adapter.render(context)

        # When SAM produced an improved query, ship that to the target AI;
        # keep the user's original text in metadata for auditability.
        original_text = context.query.text
        user_query = context.improved_query or original_text

        metadata: dict = dict(context.metadata)
        metadata.update({
            "items_included": len(context.selected),
            "items_dropped": len(context.dropped_ids),
            "dropped_ids": list(context.dropped_ids),
            "budget_total": context.budget_total,
            "budget_remaining": context.budget_remaining,
            "target_family": context.query.target.model_family.value,
            "translator_adapter": adapter.target_family_name(),
            "original_user_query": original_text,
            "query_was_improved": context.improved_query is not None
            and context.improved_query != original_text,
            "reasoning_content_included": context.reasoning_content is not None,
        })

        return TranslatedPayload(
            formatted_context=formatted,
            user_query=user_query,
            target=context.query.target,
            trace_id=context.query.trace_id,
            payload_version="1.0.0",
            orchestrator_version=orchestration.__version__,
            config_hash=config_hash,
            metadata=metadata,
        )
