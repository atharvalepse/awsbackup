import pytest

from orchestration.pipeline.contracts import (
    ModelFamily,
    OrchestrationConfig,
    TargetDescriptor,
)


def test_target_factories_cover_all_families():
    assert TargetDescriptor.for_chatgpt().model_family == ModelFamily.GPT
    assert TargetDescriptor.for_gemini().model_family == ModelFamily.GEMINI
    assert TargetDescriptor.for_claude().model_family == ModelFamily.CLAUDE
    assert TargetDescriptor.for_llama().model_family == ModelFamily.LLAMA
    assert TargetDescriptor.for_cursor("gpt-4o").model_family == ModelFamily.CURSOR


def test_target_auto_output_reserve():
    t = TargetDescriptor.for_chatgpt(context_window=100_000)
    assert t.output_reserve_tokens == 25_000


def test_config_validates_ranking_weights_sum():
    with pytest.raises(ValueError, match="sum to ~1.0"):
        OrchestrationConfig(
            ranking_weights={"semantic": 0.5, "recency": 0.3, "authority": 0.2, "pin": 0.1},
            timeouts_per_stage_ms={
                "classifier": 1, "embedder": 1, "retriever": 1, "reranker": 1,
                "sam": 1, "assembler": 1, "translator": 1,
            },
        )


def test_config_requires_all_seven_stage_timeouts():
    with pytest.raises(ValueError, match="missing required keys"):
        OrchestrationConfig(
            ranking_weights={"semantic": 0.4, "recency": 0.3, "authority": 0.2, "pin": 0.1},
            timeouts_per_stage_ms={"classifier": 1},
        )
