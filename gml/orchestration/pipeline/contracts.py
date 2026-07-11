"""Unified typed contracts passed between pipeline stages.

Single source of truth for every type a stage produces or consumes:

    Query  →  Classification  →  EmbeddedQuery  →  RetrievalHit[]  →
    RankedHit[]  →  ResolvedMemorySet  →  AssembledContext  →  TranslatedPayload

Each stage's input and output is one of these models. No stage reaches into
another stage's internals; everything that crosses a module boundary is
defined here.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Memory record (the unit of memory storage; flows through every retrieval-side stage)
# ---------------------------------------------------------------------------


class MemoryItem(BaseModel):
    """A single memory record persisted in the memory store.

    `authority_score` is the source-side trust score. `token_counts` is keyed
    by tokenizer version so cached counts invalidate when the tokenizer changes.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    summary_short: str | None = None
    summary_medium: str | None = None
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    timestamp: datetime
    source: str
    authority_score: float = Field(ge=0.0, le=1.0)
    pinned: bool = False
    # False once a newer memory supersedes this one; retrieval filters on it so
    # superseded facts don't surface on the recall path. Defaults True.
    is_latest: bool = True
    # Bitemporal validity interval (migration 013). valid_from = world time
    # the fact began to hold; valid_to = when superseded (None = currently
    # believed). None on both for rows from stores without bitemporal data.
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    # Set when this row is a chunk of a longer memory that was split on insert;
    # retrieval dedups chunks sharing a parent. None for normal atomic facts.
    parent_memory_id: str | None = None
    # Canonical entity id from write-time entity resolution (migration 014).
    # "GML" and "Gigzs Multi-LLM Layer" share one entity_id; `entity` keeps
    # the display text. None when unresolved (generic mention / pre-014 row).
    entity_id: str | None = None
    token_counts: dict[str, int] = Field(default_factory=dict)
    raw_metadata: dict = Field(default_factory=dict)
    # AAL — canonical persisted format. simplemem is a one-line sentence
    # (mirrors content for AAL-written rows). sjson is the structured
    # triple {subject, verb, object, time, negated, confidence, ...}.
    # Both are None for legacy rows written before AAL existed.
    aal_simplemem: str | None = None
    aal_sjson: dict | None = None


# ---------------------------------------------------------------------------
# Target descriptor (the destination model the pipeline is formatting for)
# ---------------------------------------------------------------------------


class ModelFamily(str, Enum):
    GPT = "gpt"
    GEMINI = "gemini"
    CLAUDE = "claude"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    CURSOR = "cursor"


class InterfaceType(str, Enum):
    API = "api"
    MCP = "mcp"
    EXTENSION = "extension"
    OTHER = "other"


class TargetDescriptor(BaseModel):
    """Describes the destination model the pipeline is preparing context for."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_family: ModelFamily
    model_version: str
    context_window: int = Field(gt=0)
    output_reserve_tokens: int | None = Field(default=None, gt=0)
    interface_type: InterfaceType
    capabilities: list[str] = Field(default_factory=list)
    cursor_backend: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _auto_output_reserve(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("output_reserve_tokens") is None:
            cw = data.get("context_window")
            if isinstance(cw, int) and cw > 0:
                data["output_reserve_tokens"] = int(cw * 0.25)
        return data

    @model_validator(mode="after")
    def _validate_invariants(self) -> "TargetDescriptor":
        if (
            self.output_reserve_tokens is not None
            and self.output_reserve_tokens >= self.context_window
        ):
            raise ValueError("output_reserve_tokens must be < context_window")
        if self.cursor_backend is not None and self.model_family != ModelFamily.CURSOR:
            raise ValueError(
                "cursor_backend may only be set when model_family is 'cursor'"
            )
        if self.model_family == ModelFamily.CURSOR and self.cursor_backend is None:
            raise ValueError(
                "cursor_backend is required when model_family is 'cursor'"
            )
        return self

    @classmethod
    def for_chatgpt(
        cls, model_version: str = "gpt-4o", context_window: int = 128_000
    ) -> "TargetDescriptor":
        return cls(
            model_family=ModelFamily.GPT,
            model_version=model_version,
            context_window=context_window,
            interface_type=InterfaceType.EXTENSION,
            capabilities=["streaming", "tool_use"],
        )

    @classmethod
    def for_gemini(
        cls, model_version: str = "gemini-2.5-pro", context_window: int = 2_000_000
    ) -> "TargetDescriptor":
        return cls(
            model_family=ModelFamily.GEMINI,
            model_version=model_version,
            context_window=context_window,
            interface_type=InterfaceType.EXTENSION,
            capabilities=["streaming", "long_context"],
        )

    @classmethod
    def for_claude(
        cls, model_version: str = "claude-opus-4-7", context_window: int = 200_000
    ) -> "TargetDescriptor":
        return cls(
            model_family=ModelFamily.CLAUDE,
            model_version=model_version,
            context_window=context_window,
            interface_type=InterfaceType.API,
            capabilities=["streaming", "tool_use", "long_context"],
        )

    @classmethod
    def for_llama(
        cls, model_version: str = "llama-3.3-70b", context_window: int = 128_000
    ) -> "TargetDescriptor":
        return cls(
            model_family=ModelFamily.LLAMA,
            model_version=model_version,
            context_window=context_window,
            interface_type=InterfaceType.API,
            capabilities=["streaming"],
        )

    @classmethod
    def for_deepseek(
        cls,
        model_version: str = "deepseek-r1:8b",
        context_window: int = 131_072,
    ) -> "TargetDescriptor":
        """DeepSeek R1 (default 8B variant served via Ollama).

        Distinct from SAM's local reasoner: that's an internal Ollama call
        SAM makes regardless of the target. A DEEPSEEK target means the
        FINAL answer also comes from a DeepSeek model.
        """
        return cls(
            model_family=ModelFamily.DEEPSEEK,
            model_version=model_version,
            context_window=context_window,
            interface_type=InterfaceType.API,
            capabilities=["streaming", "reasoning"],
        )

    @classmethod
    def for_cursor(
        cls, backend: str, context_window: int = 128_000
    ) -> "TargetDescriptor":
        return cls(
            model_family=ModelFamily.CURSOR,
            model_version=f"cursor:{backend}",
            context_window=context_window,
            interface_type=InterfaceType.EXTENSION,
            capabilities=["code_context"],
            cursor_backend=backend,
        )


# ---------------------------------------------------------------------------
# Stage I/O — Query → Classification → EmbeddedQuery
# ---------------------------------------------------------------------------


class Query(BaseModel):
    """User input and the context needed to run a pipeline pass.

    ``user_id`` scopes Postgres-backed retrieval to one tenant's memories
    via RLS. None = admin / unscoped (JSONL fallback or admin tools);
    HTTP handlers set this from ``request.state.user_id`` after auth.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    target: TargetDescriptor
    session_context: dict = Field(default_factory=dict)
    trace_id: str
    user_id: str | None = None
    # Time-travel: when set, retrieval returns the belief state AS OF this
    # instant (valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of))
    # instead of current beliefs (valid_to IS NULL). Always a parsed datetime
    # (pydantic-validated at the HTTP edge), never raw user text.
    as_of: datetime | None = None


class ClassificationSource(str, Enum):
    LLM = "llm"
    KEYWORD_FALLBACK = "keyword_fallback"
    CACHE = "cache"
    FAST_PATH = "fast_path"


class Classification(BaseModel):
    """Structured intent classification produced by the Classifier stage."""

    model_config = ConfigDict(extra="forbid")

    intent_type: str
    entities: list[str] = Field(default_factory=list)
    retrieval_hints: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    source: ClassificationSource
    degraded: bool = False


class EmbeddedQuery(BaseModel):
    """A Query+Classification carrying a dense vector for retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: Query
    classification: Classification
    vector: list[float]
    embedder_version: str


# ---------------------------------------------------------------------------
# Stage I/O — RetrievalHit → RankedHit → ResolvedMemorySet
# ---------------------------------------------------------------------------


class RetrievalHit(BaseModel):
    """One record returned by the Retriever, paired with its similarity score."""

    model_config = ConfigDict(extra="forbid")

    record: MemoryItem
    similarity: float = Field(ge=-1.0, le=1.0)


class RankedHit(BaseModel):
    """A retrieval hit augmented with per-axis Reranker scores."""

    model_config = ConfigDict(extra="forbid")

    hit: RetrievalHit
    semantic_score: float
    recency_score: float
    authority_score: float
    pin_boost: float
    final_score: float
    score_reason: str

    @property
    def record(self) -> MemoryItem:
        return self.hit.record


class ResolvedMemorySet(BaseModel):
    """Output of SAM. Either a reason-from-scratch sentinel or a filtered set
    of RankedHits with old/new conflicts resolved.

    ``improved_query`` and ``reasoning_content`` are populated when SAM's
    LLM reasoner produces them; they propagate through Assembler and
    Translator so the target AI sees both the rewritten question and the
    reasoning the local model contributed.
    """

    model_config = ConfigDict(extra="forbid")

    kept: list[RankedHit] = Field(default_factory=list)
    superseded: list[tuple[str, str]] = Field(
        default_factory=list,
        description="(loser_id, winner_id) pairs — losing record was dropped",
    )
    reason_from_scratch: bool = False
    notes: list[str] = Field(default_factory=list)
    improved_query: str | None = None
    reasoning_content: str | None = None
    reasoner_thinking: str | None = None


# ---------------------------------------------------------------------------
# Stage I/O — AssembledContext → TranslatedPayload
# ---------------------------------------------------------------------------


class AssembledContext(BaseModel):
    """Output of Assembler. Final, budget-fitted selection ready for translation.

    ``improved_query`` and ``reasoning_content`` carry SAM's LLM outputs
    through to the Translator unchanged. Translator decides whether to use
    the improved query as the user_query in the payload and where to place
    reasoning_content in the rendered template.
    """

    model_config = ConfigDict(extra="forbid")

    selected: list[RankedHit]
    query: Query
    budget_total: int
    budget_remaining: int
    dropped_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    improved_query: str | None = None
    reasoning_content: str | None = None


class TranslatedPayload(BaseModel):
    """Output of Translator. The final string + provenance ready to ship to the
    target AI."""

    model_config = ConfigDict(extra="forbid")

    formatted_context: str
    user_query: str
    target: TargetDescriptor
    trace_id: str
    payload_version: str = "1.0.0"
    orchestrator_version: str = Field(default_factory=lambda: _orchestrator_version())
    config_hash: str
    metadata: dict = Field(default_factory=dict)


def _orchestrator_version() -> str:
    from orchestration import __version__

    return __version__


# ---------------------------------------------------------------------------
# Trace + Config
# ---------------------------------------------------------------------------


class TraceEntry(BaseModel):
    """One auditable event recorded during a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    event: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict = Field(default_factory=dict)
    duration_ms: int | None = None


_REQUIRED_STAGE_KEYS = frozenset(
    {"classifier", "embedder", "retriever", "reranker", "sam", "assembler", "translator"}
)


class OrchestrationConfig(BaseModel):
    """Runtime configuration for the pipeline.

    `ranking_weights` keys are: semantic, recency, authority, pin. Must sum to ~1.0.
    `timeouts_per_stage_ms` covers all seven pipeline stages.
    `safety_margin_pct` is the per-call token margin reserved against tokenizer
    drift; clamped to [0.0, 0.5].
    """

    model_config = ConfigDict(extra="forbid")

    ranking_weights: dict[str, float]
    timeouts_per_stage_ms: dict[str, int]
    retriever_top_k: int = Field(default=50, gt=0)
    reranker_top_k: int = Field(default=10, gt=0)
    assembler_final_k: int = Field(default=5, gt=0)
    never_drop_recent_n: int = Field(default=3, ge=0)
    safety_margin_pct: float = 0.10
    recency_half_life_days: float = Field(default=30.0, gt=0.0)
    fallback_behaviors: dict = Field(default_factory=dict)

    @field_validator("ranking_weights")
    @classmethod
    def _check_ranking_weights(cls, v: dict[str, float]) -> dict[str, float]:
        expected = {"semantic", "recency", "authority", "pin"}
        missing = expected - set(v.keys())
        if missing:
            raise ValueError(f"ranking_weights missing required keys: {sorted(missing)}")
        for name, weight in v.items():
            if weight < 0:
                raise ValueError(f"ranking weight '{name}' must be non-negative, got {weight}")
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"ranking_weights must sum to ~1.0 within 0.01, got {total:.4f}"
            )
        return v

    @field_validator("timeouts_per_stage_ms")
    @classmethod
    def _check_stage_timeouts(cls, v: dict[str, int]) -> dict[str, int]:
        missing = _REQUIRED_STAGE_KEYS - set(v.keys())
        if missing:
            raise ValueError(f"timeouts_per_stage_ms missing required keys: {sorted(missing)}")
        for name, ms in v.items():
            if ms <= 0:
                raise ValueError(f"timeout '{name}' must be positive ms, got {ms}")
        return v

    @field_validator("safety_margin_pct")
    @classmethod
    def _check_safety_margin(cls, v: float) -> float:
        if not 0.0 <= v <= 0.5:
            raise ValueError(f"safety_margin_pct must be in [0.0, 0.5], got {v}")
        return v
