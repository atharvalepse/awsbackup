"""Shared fixtures for the pipeline test suite."""
import pytest

from orchestration.pipeline.contracts import (
    OrchestrationConfig,
    Query,
    TargetDescriptor,
)


@pytest.fixture
def config() -> OrchestrationConfig:
    return OrchestrationConfig(
        ranking_weights={"semantic": 0.4, "recency": 0.3, "authority": 0.2, "pin": 0.1},
        timeouts_per_stage_ms={
            "classifier": 2000,
            "embedder": 2000,
            "retriever": 3000,
            "reranker": 500,
            "sam": 500,
            "assembler": 500,
            "translator": 200,
        },
        retriever_top_k=50,
        reranker_top_k=10,
        assembler_final_k=5,
        never_drop_recent_n=3,
        safety_margin_pct=0.10,
        recency_half_life_days=30.0,
    )


@pytest.fixture
def gpt_target() -> TargetDescriptor:
    return TargetDescriptor.for_chatgpt()


@pytest.fixture
def claude_target() -> TargetDescriptor:
    return TargetDescriptor.for_claude()


@pytest.fixture
def gemini_target() -> TargetDescriptor:
    return TargetDescriptor.for_gemini()


@pytest.fixture
def llama_target() -> TargetDescriptor:
    return TargetDescriptor.for_llama()


@pytest.fixture
def deepseek_target() -> TargetDescriptor:
    return TargetDescriptor.for_deepseek()


def make_query(text: str, target: TargetDescriptor, trace_id: str = "trace-test") -> Query:
    return Query(text=text, target=target, session_context={}, trace_id=trace_id)
