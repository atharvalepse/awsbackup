"""Gemini-backed Classifier with a keyword-fallback safety net.

Pipeline order inside ``classify``: cache → Gemini Flash LLM → keyword fallback.
The pleasantry short-circuit is intentionally NOT here — that's a flow
decision and lives in the top-level Pipeline.
"""
import asyncio
import hashlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from orchestration.classifier.base import Classifier
from orchestration.classifier.keyword_classifier import _match_keyword
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import Classification, ClassificationSource, Query


slog = StructuredLogger("classifier")

_INTENT_TYPES = ("coding", "debugging", "writing", "research", "question", "task", "other")
DEFAULT_MODEL = "gemini-flash-latest"
# Matches the [timeouts_per_stage_ms].classifier default (2000 ms). The
# pipeline additionally enforces its configured stage timeout via wait_for.
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60


class _IntentLLMSchema(BaseModel):
    """Shape the LLM is asked to return — distinct from Classification so
    LLM responses stay free of orchestrator-internal fields like ``source``."""

    model_config = ConfigDict(extra="ignore")

    intent_type: str
    entities: list[str] = Field(default_factory=list)
    retrieval_hints: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("intent_type")
    @classmethod
    def _coerce_intent(cls, v: str) -> str:
        # The LLM occasionally hallucinates labels outside the taxonomy;
        # downstream consumers assume one of _INTENT_TYPES, so coerce.
        norm = v.strip().lower()
        return norm if norm in _INTENT_TYPES else "other"


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> Classification | None: ...

    @abstractmethod
    async def set(self, key: str, value: Classification, ttl_seconds: int) -> None: ...


@dataclass
class _CacheEntry:
    value: Classification
    expiry: float


class InMemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}

    async def get(self, key: str) -> Classification | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() >= entry.expiry:
            self._store.pop(key, None)
            return None
        return entry.value

    async def set(self, key: str, value: Classification, ttl_seconds: int) -> None:
        self._store[key] = _CacheEntry(value=value, expiry=time.time() + ttl_seconds)


def _cache_key(text: str) -> str:
    # Classification depends only on the query text, so the key must not
    # fragment by session — that made the hit rate near-zero multi-tenant.
    # Normalize whitespace + case so trivial paraphrases share an entry.
    norm = " ".join(text.split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _build_prompt(text: str) -> str:
    return (
        "You are an intent classifier for a memory orchestration system.\n"
        "Classify the user query and respond with a JSON object containing EXACTLY these keys:\n"
        f'  - "intent_type": one of: {", ".join(_INTENT_TYPES)}\n'
        '  - "entities": array of strings (concrete nouns: services, tools, files, people, etc.)\n'
        '  - "retrieval_hints": object with additional retrieval signals (use {} if none)\n'
        '  - "confidence": number between 0.0 and 1.0\n'
        "\n"
        "Return JSON only — no prose, no markdown fences, no extra keys.\n\n"
        f"User query: {text!r}\n"
    )


class LLMClassifier(Classifier):
    """Gemini-Flash-backed Classifier with cache + keyword fallback.

    When ``api_key`` and ``GEMINI_API_KEY`` are both unset, runs in stub mode
    (keyword fallback only, ``degraded=False``).

    Fast-path mode (default, ``GML_CLASSIFIER_FAST=0`` to disable): a cache
    miss returns the keyword classification immediately (source=FAST_PATH)
    and refines via the LLM in a background task that warms the cache for
    subsequent identical queries. This removes the 200-500 ms external
    round-trip from the request hot path — intent is consumed by the SAM
    prompt and the embedder signal, neither of which justifies blocking
    the whole pipeline on a network call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cache: CacheBackend | None = None,
        fast_path: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.cache: CacheBackend = cache if cache is not None else InMemoryCache()
        self._client = None  # genai.Client, built lazily and reused across calls
        self._stub_mode = self.api_key is None
        if fast_path is None:
            fast_path = os.environ.get("GML_CLASSIFIER_FAST", "1") == "1"
        self.fast_path = fast_path
        # Background refinement bookkeeping: dedupe in-flight keys and hold
        # strong task refs so fire-and-forget tasks aren't GC'd mid-flight.
        self._inflight: set[str] = set()
        self._refine_tasks: set[asyncio.Task] = set()
        if self._stub_mode:
            slog.warning(
                event="gemini_api_key_missing_stub_mode",
                reason="GEMINI_API_KEY not set; using keyword fallback",
                degraded_mode=True,
            )

    async def classify(self, query: Query) -> Classification:
        text = query.text

        if self._stub_mode:
            return _match_keyword(text)

        key = _cache_key(text)
        cached = await self.cache.get(key)
        if cached is not None:
            return cached.model_copy(update={"source": ClassificationSource.CACHE})

        if self.fast_path:
            self._spawn_refine(key, text, trace_id=query.trace_id)
            return _match_keyword(text).model_copy(
                update={"source": ClassificationSource.FAST_PATH}
            )

        try:
            llm_result = await self._call_llm(text)
        except asyncio.TimeoutError:
            slog.warning(
                event="llm_timeout_keyword_fallback",
                trace_id=query.trace_id,
                timeout_seconds=self.timeout_seconds,
                model=self.model,
                degraded_mode=True,
            )
            return _match_keyword(text, degraded=True)
        except (ValueError, ValidationError) as exc:
            slog.warning(
                event="llm_malformed_output_keyword_fallback",
                trace_id=query.trace_id,
                error=str(exc),
                model=self.model,
                degraded_mode=True,
            )
            return _match_keyword(text, degraded=True)
        except Exception as exc:
            slog.warning(
                event="llm_call_failed_keyword_fallback",
                trace_id=query.trace_id,
                error_type=type(exc).__name__,
                error=str(exc),
                model=self.model,
                degraded_mode=True,
            )
            return _match_keyword(text, degraded=True)

        classification = Classification(
            intent_type=llm_result.intent_type,
            entities=llm_result.entities,
            retrieval_hints=llm_result.retrieval_hints,
            confidence=llm_result.confidence,
            source=ClassificationSource.LLM,
        )
        await self.cache.set(key, classification, ttl_seconds=DEFAULT_CACHE_TTL_SECONDS)
        return classification

    def _spawn_refine(self, key: str, text: str, trace_id: str | None) -> None:
        """Warm the cache with the LLM result off the hot path. At most one
        in-flight refinement per cache key."""
        if key in self._inflight:
            return
        self._inflight.add(key)
        task = asyncio.create_task(self._refine(key, text, trace_id))
        self._refine_tasks.add(task)
        task.add_done_callback(self._refine_tasks.discard)

    async def _refine(self, key: str, text: str, trace_id: str | None) -> None:
        try:
            llm_result = await self._call_llm(text)
            classification = Classification(
                intent_type=llm_result.intent_type,
                entities=llm_result.entities,
                retrieval_hints=llm_result.retrieval_hints,
                confidence=llm_result.confidence,
                source=ClassificationSource.LLM,
            )
            await self.cache.set(
                key, classification, ttl_seconds=DEFAULT_CACHE_TTL_SECONDS
            )
        except Exception as exc:
            # Background-only failure: the request already got the keyword
            # answer, so just log — the next identical query retries.
            slog.warning(
                event="classifier_background_refine_failed",
                trace_id=trace_id,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
                model=self.model,
            )
        finally:
            self._inflight.discard(key)

    async def _call_llm(self, text: str) -> _IntentLLMSchema:
        from google import genai
        from google.genai import types as genai_types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        client = self._client
        prompt = _build_prompt(text)
        config = genai_types.GenerateContentConfig(response_mime_type="application/json")

        async def _do_call():
            return await client.aio.models.generate_content(
                model=self.model, contents=[prompt], config=config
            )

        response = await asyncio.wait_for(_do_call(), timeout=self.timeout_seconds)

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, _IntentLLMSchema):
            return parsed
        if isinstance(parsed, dict):
            return _IntentLLMSchema.model_validate(parsed)

        text_out = getattr(response, "text", None)
        if text_out:
            return _IntentLLMSchema.model_validate_json(text_out)

        raise ValueError("LLM returned an empty response with no parseable content")
