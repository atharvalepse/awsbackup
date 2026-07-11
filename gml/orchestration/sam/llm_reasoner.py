"""LLMReasoner — wraps prompt templates + JSON parsing for SAM's LLM jobs.

Two methods, one per SAM call site:

* ``reason_for_empty_memory(query, classification)`` — emits a rewritten
  query and reasoning content for the NO-memory branch.

* ``reason_over_conflicts(query, ranked)`` — emits drop decisions plus a
  rewritten query and reasoning content for the post-rerank branch.

Both return :class:`LLMReasoningResult`. Errors from the underlying client
or JSON parsing surface as :class:`LLMReasonerError` — SAM catches and
falls back to the heuristic path.
"""
import json
from dataclasses import dataclass, field

from orchestration.errors import SAMError
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import Classification, Query, RankedHit
from orchestration.sam._ollama_client import OllamaClient


slog = StructuredLogger("sam.reasoner")


class LLMReasonerError(SAMError):
    """LLM call or output parsing failed; SAM should fall back."""


@dataclass(frozen=True)
class LLMReasoningResult:
    """Structured output of an LLMReasoner call.

    ``thinking`` is the model's ``<think>`` content (empty for non-R1 models).
    ``improved_query`` is None when the model declined to rewrite. ``drop_ids``
    is empty for the reason-from-scratch case.
    """

    improved_query: str | None
    reasoning_content: str | None
    drop_ids: list[str] = field(default_factory=list)
    thinking: str = ""


_REASON_FROM_SCRATCH_PROMPT = """\
You are SAM, a query-improvement and reasoning layer for a memory-augmented
AI assistant. The user has asked a question and NO prior memory is available
to ground the answer.

Your job: produce (a) a clearer, more specific rewrite of the user's question
that the assistant can answer well, and (b) optional reasoning content.

CRITICAL — you have NO stored facts about this user, their systems, projects,
people, versions, dates, or data. Do NOT invent any. Do NOT state any specific
fact in the rewrite or the reasoning. The rewrite must only clarify wording —
never add details that weren't in the original question. Because there is no
memory to ground anything, the "reasoning" should normally be empty (""); only
note, at most, that no relevant memory was found.

User query: {query_text!r}
Classified intent: {intent_type}
Extracted entities: {entities}

Respond with JSON ONLY, no markdown fences, no prose around the JSON. Exact shape:

{{
  "improved_query": "<a wording-only clarification of the user's question; add NO new facts>",
  "reasoning": "<empty string, or at most one sentence noting no memory was found — never any invented fact>"
}}
"""


_RESOLVE_CONFLICTS_PROMPT = """\
You are SAM, a query-improvement and reasoning layer for a memory-augmented
AI assistant. The user asked a question and the system retrieved memories
that may be relevant. Your default is to KEEP every retrieved memory and
let the downstream AI decide what's useful — the retrieval pipeline already
ranked these for relevance.

Only drop a memory if it meets ALL THREE conditions:
  (a) Another memory in the list has the SAME entity AND attribute,
      AND a DIFFERENT value (an explicit contradiction).
  (b) The other memory's timestamp is strictly newer.
  (c) Both memories are about the same concrete fact (not loosely related).

If you're unsure, KEEP IT. Never drop a memory just because it seems less
relevant to the query — relevance ranking already happened. Never drop a
memory because the entity/attribute is "different topic" from the query.
Drops are ONLY for explicit value-contradictions.

In nearly all cases the correct answer is `"drop_ids": []`. Dropping more
than HALF of the input list is almost always wrong — when in doubt, drop
zero.

User query: {query_text!r}

Retrieved memories (sorted by ranking score, highest first):
{memory_list}

GROUNDING — the "reasoning" must use ONLY facts that appear verbatim in the
retrieved memories above. Do NOT introduce any name, number, version, date, or
detail that is not in those memories. If the memories don't actually address
the user's query, say exactly that in the reasoning rather than guessing. Never
fabricate.

Respond with JSON ONLY, no markdown fences, no prose around the JSON. Exact shape:

{{
  "drop_ids": ["<id_of_strictly_contradicted_memory>", ...],
  "improved_query": "<a wording-only clarification informed by the memories, or the original text — add NO facts absent from the memories>",
  "reasoning": "<1-3 sentences that only restate/connect facts present in the memories above; if they don't address the query, say so. Never invent.>"
}}
"""


def _format_memory_line(rh: RankedHit) -> str:
    rec = rh.record
    parts = [f"- id={rec.id!r}", f"timestamp={rec.timestamp.isoformat()}"]
    if rec.entity:
        parts.append(f"entity={rec.entity!r}")
    if rec.attribute:
        parts.append(f"attribute={rec.attribute!r}")
    if rec.value:
        parts.append(f"value={rec.value!r}")
    parts.append(f"score={rh.final_score:.2f}")
    parts.append(f"content={rec.content!r}")
    return "  ".join(parts)


def _parse_json_answer(answer: str) -> dict:
    """Try to extract a JSON object from ``answer``. Tolerant of leading/trailing
    junk by locating the outermost braces."""
    text = answer.strip()
    if not text:
        raise LLMReasonerError("LLM returned an empty answer")
    # Strip Markdown fences just in case the model ignored the instruction.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Take outermost { ... } substring
    open_idx = text.find("{")
    close_idx = text.rfind("}")
    if open_idx < 0 or close_idx <= open_idx:
        raise LLMReasonerError(f"LLM answer contained no JSON object: {answer!r}")
    candidate = text[open_idx : close_idx + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMReasonerError(f"LLM JSON parse failed: {exc}; answer={answer!r}") from exc


class LLMReasoner:
    """Wraps a :class:`OllamaClient` with SAM-shaped prompts + parsers."""

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    async def reason_for_empty_memory(
        self, query: Query, classification: Classification
    ) -> LLMReasoningResult:
        prompt = _REASON_FROM_SCRATCH_PROMPT.format(
            query_text=query.text,
            intent_type=classification.intent_type,
            entities=", ".join(classification.entities) or "(none)",
        )
        try:
            gen = await self.client.generate(prompt, json_mode=True, temperature=0.0)
        except Exception as exc:
            raise LLMReasonerError(
                f"Ollama generate failed: {type(exc).__name__}: {exc}"
            ) from exc

        data = _parse_json_answer(gen.answer)
        improved = (data.get("improved_query") or "").strip() or None
        reasoning = (data.get("reasoning") or "").strip() or None
        return LLMReasoningResult(
            improved_query=improved,
            reasoning_content=reasoning,
            drop_ids=[],
            thinking=gen.thinking,
        )

    async def reason_over_conflicts(
        self, query: Query, ranked: list[RankedHit]
    ) -> LLMReasoningResult:
        if not ranked:
            return LLMReasoningResult(
                improved_query=None,
                reasoning_content=None,
                drop_ids=[],
                thinking="",
            )

        memory_list = "\n".join(_format_memory_line(rh) for rh in ranked)
        prompt = _RESOLVE_CONFLICTS_PROMPT.format(
            query_text=query.text,
            memory_list=memory_list,
        )
        try:
            gen = await self.client.generate(prompt, json_mode=True, temperature=0.0)
        except Exception as exc:
            raise LLMReasonerError(
                f"Ollama generate failed: {type(exc).__name__}: {exc}"
            ) from exc

        data = _parse_json_answer(gen.answer)
        improved = (data.get("improved_query") or "").strip() or None
        reasoning = (data.get("reasoning") or "").strip() or None
        raw_drops = data.get("drop_ids") or []
        if not isinstance(raw_drops, list):
            raise LLMReasonerError(f"drop_ids must be a list, got {type(raw_drops).__name__}")
        # Defensive: only keep IDs that actually exist in `ranked`
        valid_ids = {rh.record.id for rh in ranked}
        drop_ids = [str(x) for x in raw_drops if str(x) in valid_ids]
        return LLMReasoningResult(
            improved_query=improved,
            reasoning_content=reasoning,
            drop_ids=drop_ids,
            thinking=gen.thinking,
        )
