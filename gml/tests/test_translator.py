import pytest

from orchestration.pipeline.contracts import AssembledContext, ModelFamily
from orchestration.translator import (
    ClaudeAdapter,
    DeepSeekAdapter,
    GeminiAdapter,
    GPTAdapter,
    LlamaAdapter,
    Translator,
)

from tests.conftest import make_query


def _empty_context(target):
    return AssembledContext(
        selected=[],
        query=make_query("hello world", target),
        budget_total=1000,
        budget_remaining=1000,
        dropped_ids=[],
        metadata={"reason_from_scratch": False},
    )


def test_translator_dispatches_by_family(gpt_target, claude_target, gemini_target, llama_target, deepseek_target):
    t = Translator()
    assert t.adapter_for(gpt_target).target_family_name() == "gpt"
    assert t.adapter_for(claude_target).target_family_name() == "claude"
    assert t.adapter_for(gemini_target).target_family_name() == "gemini"
    assert t.adapter_for(llama_target).target_family_name() == "llama"
    assert t.adapter_for(deepseek_target).target_family_name() == "deepseek"


def test_deepseek_adapter_renders(deepseek_target):
    out = DeepSeekAdapter().render(_empty_context(deepseek_target))
    assert "Retrieved Context" in out
    assert "User Query" in out


def test_cursor_dispatches_to_gpt():
    from orchestration.pipeline.contracts import TargetDescriptor
    cursor = TargetDescriptor.for_cursor("gpt-4o")
    t = Translator()
    assert t.adapter_for(cursor).target_family_name() == "gpt"


def test_gpt_adapter_renders_empty_template_path(gpt_target):
    output = GPTAdapter().render(_empty_context(gpt_target))
    assert "Context" in output
    assert "no relevant context retrieved" in output.lower()


def test_claude_adapter_uses_xml_tags(claude_target):
    out = ClaudeAdapter().render(_empty_context(claude_target))
    assert "<context>" in out


def test_gemini_adapter_uses_markdown_heading(gemini_target):
    out = GeminiAdapter().render(_empty_context(gemini_target))
    assert "Retrieved Context" in out


def test_llama_adapter_uses_role_sections(llama_target):
    out = LlamaAdapter().render(_empty_context(llama_target))
    assert "### Context" in out
    assert "### User Query" in out


def test_translate_produces_translated_payload(gpt_target):
    t = Translator()
    payload = t.translate(_empty_context(gpt_target), config_hash="abc123")
    assert payload.config_hash == "abc123"
    assert payload.user_query == "hello world"
    assert payload.target.model_family == ModelFamily.GPT
    assert payload.metadata["items_included"] == 0


def test_reason_from_scratch_renders_special_line(gpt_target):
    ctx = AssembledContext(
        selected=[],
        query=make_query("q", gpt_target),
        budget_total=1000, budget_remaining=1000,
        dropped_ids=[],
        metadata={"reason_from_scratch": True},
    )
    output = GPTAdapter().render(ctx)
    assert "reason from scratch" in output.lower()


def test_reasoning_content_renders_in_all_adapters(
    gpt_target, gemini_target, claude_target, llama_target, deepseek_target
):
    """SAM reasoning content shows up under a SAM Reasoning heading in every target."""
    def ctx_for(target):
        return AssembledContext(
            selected=[],
            query=make_query("q", target),
            budget_total=1000, budget_remaining=1000,
            dropped_ids=[],
            metadata={"reason_from_scratch": True},
            reasoning_content="key insight from DeepSeek R1",
        )

    assert "key insight from DeepSeek R1" in GPTAdapter().render(ctx_for(gpt_target))
    assert "key insight from DeepSeek R1" in GeminiAdapter().render(ctx_for(gemini_target))
    assert "key insight from DeepSeek R1" in ClaudeAdapter().render(ctx_for(claude_target))
    assert "key insight from DeepSeek R1" in LlamaAdapter().render(ctx_for(llama_target))
    assert "key insight from DeepSeek R1" in DeepSeekAdapter().render(ctx_for(deepseek_target))


def test_translate_uses_improved_query_when_present(gpt_target):
    ctx = AssembledContext(
        selected=[],
        query=make_query("vague q", gpt_target),
        budget_total=1000, budget_remaining=1000,
        dropped_ids=[],
        metadata={"reason_from_scratch": True},
        improved_query="much more precise question",
    )
    payload = Translator().translate(ctx, config_hash="h")
    assert payload.user_query == "much more precise question"
    assert payload.metadata["original_user_query"] == "vague q"
    assert payload.metadata["query_was_improved"] is True
