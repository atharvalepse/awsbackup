"""SAM — semantic alignment module.

Called by the Pipeline in exactly two cases:

1. ``reason_from_scratch(query, classification)`` — Retriever returned
   nothing. SAM uses its LLM reasoner (default: local DeepSeek R1 8B via
   Ollama) to produce an improved version of the user's question plus
   reasoning content the target AI should consider while answering.

2. ``resolve_conflicts(query, ranked)`` — Reranker returned candidates.
   SAM uses the same reasoner to (a) decide which retrieved memories are
   superseded by newer ones and drop them, (b) rewrite the user's query
   informed by the retrieved memories, (c) emit reasoning content.

When the LLM reasoner is absent or fails (Ollama down, parse error,
timeout), SAM falls back to the heuristic entity-attribute resolver. The
heuristic path cannot produce improved_query or reasoning_content; those
fields stay None.

Routing modes (GML_SAM_MODE, or the ``mode`` constructor arg):

* ``heuristic_first`` (default) — resolve_conflicts runs the cheap
  entity-attribute resolver first and only escalates to the LLM when the
  heuristic ABSTAINS: contradicting values survive in the kept set,
  write-gate conflict links point inside the kept set, or the resolver
  errored. A local reasoning model on the critical path of every query
  is where the synthesize-latency budget went; escalation keeps it for
  the cases that actually need judgment.
* ``llm_first`` — legacy behavior: LLM whenever a reasoner is wired,
  heuristic only as the failure fallback.
"""
import os

from orchestration.observability.logging import StructuredLogger
from orchestration.observability.metrics import (
    ORCHESTRATION_CONFLICTS_DETECTED_TOTAL,
    ORCHESTRATION_FAILURES_TOTAL,
)
from orchestration.pipeline.contracts import (
    Classification,
    Query,
    RankedHit,
    ResolvedMemorySet,
)
from orchestration.sam._ollama_client import (
    DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL,
    HTTPOllamaClient,
    make_local_llm_client,
)
from orchestration.sam.llm_reasoner import LLMReasoner, LLMReasonerError
from orchestration.sam.resolvers.base import ConflictResolver
from orchestration.sam.resolvers.heuristic import HeuristicConflictResolver


slog = StructuredLogger("sam")

DEFAULT_DROP_THRESHOLD = 0.8


class SAM:
    """Two-mode reasoning module: LLM-backed with a heuristic safety net.

    ``reasoner``: optional :class:`LLMReasoner`. When set, both methods route
    through it. When None, the heuristic resolver handles ``resolve_conflicts``
    and ``reason_from_scratch`` returns a sentinel.

    ``conflict_resolver``: heuristic fallback used when ``reasoner`` is None
    or raises :class:`LLMReasonerError`.

    Example:
        >>> sam = SAM.with_ollama()                  # local DeepSeek R1 8B
        >>> sam_heuristic_only = SAM(reasoner=None)  # no LLM
    """

    def __init__(
        self,
        reasoner: LLMReasoner | None = None,
        conflict_resolver: ConflictResolver | None = None,
        drop_threshold: float = DEFAULT_DROP_THRESHOLD,
        mode: str | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.conflict_resolver = conflict_resolver or HeuristicConflictResolver()
        if not 0.0 <= drop_threshold <= 1.0:
            raise ValueError("drop_threshold must be in [0.0, 1.0]")
        self.drop_threshold = drop_threshold
        self.mode = (
            mode or os.environ.get("GML_SAM_MODE", "heuristic_first")
        ).strip().lower()

    @classmethod
    def with_ollama(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        # Bumped from 15s to 90s. llama.cpp serializes inference; the 15s
        # cap was firing on queued requests before they could be processed.
        # 90s is the cap when SAM is on the critical path of every query;
        # the SAM-skip optimization keeps actual SAM invocations rare.
        timeout_seconds: float = 90.0,
        conflict_resolver: ConflictResolver | None = None,
        drop_threshold: float = DEFAULT_DROP_THRESHOLD,
        backend: str | None = None,
        mode: str | None = None,
    ) -> "SAM":
        """Build a SAM wired to a local LLM (llama.cpp or Ollama).

        Backend chosen via ``GML_LLM_BACKEND`` env var (default ``llamacpp``).
        Both backends fall back to the heuristic resolver if the server is
        unreachable or returns un-parseable output.

        The method name is kept (``with_ollama``) for backwards compatibility
        — it accepts either backend now.
        """
        client = make_local_llm_client(
            backend=backend,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        return cls(
            reasoner=LLMReasoner(client),
            conflict_resolver=conflict_resolver,
            drop_threshold=drop_threshold,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # NO-branch entry point
    # ------------------------------------------------------------------

    async def reason_from_scratch(
        self, query: Query, classification: Classification
    ) -> ResolvedMemorySet:
        slog.info(
            event="reason_from_scratch",
            trace_id=query.trace_id,
            intent_type=classification.intent_type,
            llm_enabled=self.reasoner is not None,
        )

        if self.reasoner is None:
            return ResolvedMemorySet(
                kept=[],
                superseded=[],
                reason_from_scratch=True,
                notes=[
                    "no relevant memory retrieved; target AI should reason from scratch"
                ],
            )

        try:
            result = await self.reasoner.reason_for_empty_memory(query, classification)
        except LLMReasonerError as exc:
            ORCHESTRATION_FAILURES_TOTAL.inc(stage="sam_reasoner")
            slog.warning(
                event="reasoner_failed_heuristic_fallback",
                trace_id=query.trace_id,
                error=str(exc),
                degraded_mode=True,
            )
            return ResolvedMemorySet(
                kept=[],
                superseded=[],
                reason_from_scratch=True,
                notes=[
                    "no relevant memory retrieved; SAM LLM reasoner unavailable",
                    f"reasoner_error: {exc}",
                ],
            )

        return ResolvedMemorySet(
            kept=[],
            superseded=[],
            reason_from_scratch=True,
            notes=["no relevant memory retrieved; SAM reasoned over query"],
            improved_query=result.improved_query,
            reasoning_content=result.reasoning_content,
            reasoner_thinking=result.thinking or None,
        )

    # ------------------------------------------------------------------
    # YES-branch entry point
    # ------------------------------------------------------------------

    async def resolve_conflicts(
        self, query: Query, ranked: list[RankedHit]
    ) -> ResolvedMemorySet:
        if not ranked:
            return ResolvedMemorySet(kept=[], superseded=[], reason_from_scratch=False)

        if self.reasoner is not None and self.mode == "llm_first":
            try:
                return await self._resolve_via_llm(query, ranked)
            except LLMReasonerError as exc:
                ORCHESTRATION_FAILURES_TOTAL.inc(stage="sam_reasoner")
                slog.warning(
                    event="reasoner_failed_heuristic_fallback",
                    trace_id=query.trace_id,
                    error=str(exc),
                    degraded_mode=True,
                )
                # fall through to heuristic

        resolved = await self._resolve_via_heuristic(query, ranked)

        if self.reasoner is None or self.mode == "llm_first":
            return resolved

        abstained = self._heuristic_abstention(resolved)
        if abstained is None:
            return resolved

        slog.info(
            event="sam_escalate_to_llm",
            trace_id=query.trace_id,
            reason=abstained,
            kept=len(resolved.kept),
        )
        try:
            return await self._resolve_via_llm(query, ranked)
        except LLMReasonerError as exc:
            ORCHESTRATION_FAILURES_TOTAL.inc(stage="sam_reasoner")
            slog.warning(
                event="reasoner_failed_keeping_heuristic_result",
                trace_id=query.trace_id,
                error=str(exc),
                degraded_mode=True,
            )
            return resolved

    def _heuristic_abstention(self, resolved: ResolvedMemorySet) -> str | None:
        """Did the heuristic leave a contradiction standing that needs LLM
        judgment? Returns the escalation reason, or None when the cheap
        result is safe to ship."""
        for note in resolved.notes or []:
            if note.startswith("conflict resolution skipped"):
                return "resolver_error"

        # Distinct values surviving within one (entity, attribute) group —
        # e.g. equal timestamps gave the age heuristic nothing to penalize.
        groups: dict[tuple[str, str], set[str]] = {}
        for rh in resolved.kept:
            rec = rh.record
            if rec.entity and rec.attribute:
                key = (rec.entity.strip().lower(), rec.attribute.strip().lower())
                groups.setdefault(key, set()).add((rec.value or "").strip().lower())
        for (entity, attribute), values in groups.items():
            if len(values) > 1:
                return f"unresolved_values:{entity}/{attribute}"

        # Write-gate conflict links pointing INSIDE the kept set: the store
        # flagged these as contradicting at ingest, and both still surface.
        kept_ids = {rh.record.id for rh in resolved.kept}
        for rh in resolved.kept:
            partners = set(
                (rh.record.raw_metadata or {}).get("conflict_with") or []
            )
            if partners & kept_ids:
                return "conflict_links_in_kept"
        return None

    # ------------------------------------------------------------------
    # Internal: LLM and heuristic paths
    # ------------------------------------------------------------------

    async def _resolve_via_llm(
        self, query: Query, ranked: list[RankedHit]
    ) -> ResolvedMemorySet:
        result = await self.reasoner.reason_over_conflicts(query, ranked)

        drop_set = set(result.drop_ids)

        # Safety net 1: an over-eager LLM that drops >50% of inputs is
        # almost certainly hallucinating contradictions. Ignore its drop
        # list and keep everything.
        if len(drop_set) > len(ranked) // 2:
            slog.warning(
                event="sam_llm_overdrop_ignored",
                trace_id=query.trace_id,
                drop_count=len(drop_set),
                input_count=len(ranked),
                drop_ids=sorted(drop_set),
            )
            drop_set = set()

        # Safety net 2: reject any drop where the dropped memory is NOT
        # strictly older than another kept memory with the same entity +
        # attribute. The prompt forbids this, but smaller models sometimes
        # drop the wrong direction (e.g. drop the newer answer). Catch and
        # discard those drops — better to over-keep than to remove the
        # right memory.
        by_id = {rh.record.id: rh for rh in ranked}
        rejected_drops: list[str] = []
        validated_drops: set = set()
        for drop_id in drop_set:
            rh_dropped = by_id.get(drop_id)
            if rh_dropped is None:
                continue
            rec_d = rh_dropped.record
            if not rec_d.entity or not rec_d.attribute:
                # LLM claims contradiction but no entity/attribute to compare
                rejected_drops.append(drop_id)
                continue
            # Find any other ranked memory with same (entity, attribute) and a strictly-newer timestamp
            has_newer_peer = any(
                rh.record.id != drop_id
                and rh.record.entity == rec_d.entity
                and rh.record.attribute == rec_d.attribute
                and rh.record.timestamp > rec_d.timestamp
                for rh in ranked
            )
            if has_newer_peer:
                validated_drops.add(drop_id)
            else:
                rejected_drops.append(drop_id)

        if rejected_drops:
            slog.warning(
                event="sam_llm_invalid_drops_rejected",
                trace_id=query.trace_id,
                rejected=rejected_drops,
                kept_instead=True,
            )
        drop_set = validated_drops

        kept = [rh for rh in ranked if rh.record.id not in drop_set]

        # Best-effort supersession pairs: for each dropped id, name the
        # highest-scoring kept candidate that shares entity+attribute.
        superseded: list[tuple[str, str]] = []
        for rh in ranked:
            if rh.record.id not in drop_set:
                continue
            winner = self._find_winner(rh, kept)
            if winner is not None:
                superseded.append((rh.record.id, winner))

        if superseded:
            ORCHESTRATION_CONFLICTS_DETECTED_TOTAL.inc(amount=len(superseded))

        notes: list[str] = []
        if drop_set:
            notes.append(f"LLM dropped {len(drop_set)} superseded record(s): {sorted(drop_set)}")
        else:
            notes.append("LLM reviewed memories; no supersessions found")

        return ResolvedMemorySet(
            kept=kept,
            superseded=superseded,
            reason_from_scratch=False,
            notes=notes,
            improved_query=result.improved_query,
            reasoning_content=result.reasoning_content,
            reasoner_thinking=result.thinking or None,
        )

    async def _resolve_via_heuristic(
        self, query: Query, ranked: list[RankedHit]
    ) -> ResolvedMemorySet:
        records = [rh.record for rh in ranked]
        try:
            penalties, supersessions = await self.conflict_resolver.score_conflicts(records)
        except Exception as exc:
            slog.warning(
                event="conflict_resolver_failed",
                trace_id=query.trace_id,
                error_type=type(exc).__name__,
                error=str(exc),
                degraded_mode=True,
            )
            return ResolvedMemorySet(
                kept=ranked,
                superseded=[],
                reason_from_scratch=False,
                notes=[f"conflict resolution skipped: {type(exc).__name__}"],
            )

        if supersessions:
            ORCHESTRATION_CONFLICTS_DETECTED_TOTAL.inc(amount=len(supersessions))

        kept: list[RankedHit] = []
        dropped_count = 0
        for rh in ranked:
            penalty = penalties.get(rh.record.id, 0.0)
            if penalty >= self.drop_threshold:
                dropped_count += 1
                continue
            kept.append(rh)

        notes: list[str] = []
        if dropped_count:
            notes.append(f"heuristic dropped {dropped_count} record(s) as superseded")

        return ResolvedMemorySet(
            kept=kept,
            superseded=supersessions,
            reason_from_scratch=False,
            notes=notes,
        )

    @staticmethod
    def _find_winner(loser: RankedHit, kept: list[RankedHit]) -> str | None:
        """Return the id of the highest-scoring kept hit sharing entity+attribute
        with ``loser``, or None if no structural match exists."""
        if loser.record.entity is None or loser.record.attribute is None:
            return None
        candidates = [
            k for k in kept
            if k.record.entity == loser.record.entity
            and k.record.attribute == loser.record.attribute
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda k: k.final_score).record.id
