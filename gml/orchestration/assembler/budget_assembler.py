"""Budget-aware Assembler with per-item compression.

Walks the resolved RankedHits in score order, fitting each into the remaining
budget — full content first, then ``summary_medium``, then ``summary_short``,
then dropped. Pinned + most-recent-N items are protected and tried first;
they only fall back to summaries or get dropped when the budget is too tight
to fit them at full.

Decoupled from Translator: the Pipeline computes the empty-template overhead
once against the chosen adapter and passes it into ``package``.
"""
import hashlib
from collections import OrderedDict
from typing import Callable
from datetime import datetime, timezone

from orchestration.assembler.base import Assembler
from orchestration.errors import BudgetExceededError
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import (
    AssembledContext,
    MemoryItem,
    OrchestrationConfig,
    Query,
    RankedHit,
    ResolvedMemorySet,
)
from orchestration.tokenizers.base import Tokenizer


slog = StructuredLogger("assembler")

_CACHE_MAX_SIZE = 10_000


class _LRUCache:
    def __init__(self, maxsize: int = _CACHE_MAX_SIZE) -> None:
        self._d: "OrderedDict[tuple, int]" = OrderedDict()
        self._max = maxsize

    def get(self, key: tuple) -> int | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def set(self, key: tuple, value: int) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        elif len(self._d) >= self._max:
            self._d.popitem(last=False)
        self._d[key] = value


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


class BudgetAssembler(Assembler):
    """Greedy budget-fitting Assembler.

    Behavior:
    1. Compute ``memory_budget`` = context_window - output_reserve - query_tokens
       - template_overhead - safety_margin. If <= 0, raise BudgetExceededError.
    2. If ``resolved.reason_from_scratch``, return an empty context (no fitting).
    3. Take top-``final`` of ``resolved.kept`` plus all protected items
       (pinned ∪ recent-N).
    4. Greedily fit each (full → medium → short → dropped).
    5. Return AssembledContext sorted by final_score desc.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: OrchestrationConfig,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.now_provider = now_provider or _default_now
        self._lru = _LRUCache()
        self.cache_hits = 0           # LRU hits — token count served from in-process cache
        self.precomputed_hits = 0     # MemoryItem.token_counts[version] hit (no tokenizer call)
        self.cache_misses = 0         # neither LRU nor precomputed — full tokenize ran

    # -- token counting ------------------------------------------------------

    def _count_tokens(
        self, item_id: str, content: str, precomputed: int | None = None
    ) -> int:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        key = (item_id, self.tokenizer.version, content_hash)
        cached = self._lru.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if precomputed is not None:
            self.precomputed_hits += 1
            self._lru.set(key, precomputed)
            return precomputed
        self.cache_misses += 1
        n = self.tokenizer.count(content)
        self._lru.set(key, n)
        return n

    def _count_full(self, item: MemoryItem) -> int:
        return self._count_tokens(
            item.id, item.content, precomputed=item.token_counts.get(self.tokenizer.version)
        )

    def _count_medium(self, item: MemoryItem) -> int:
        assert item.summary_medium is not None
        return self._count_tokens(f"{item.id}:medium", item.summary_medium)

    def _count_short(self, item: MemoryItem) -> int:
        assert item.summary_short is not None
        return self._count_tokens(f"{item.id}:short", item.summary_short)

    # -- public API ----------------------------------------------------------

    def package(
        self,
        resolved: ResolvedMemorySet,
        query: Query,
        template_overhead_tokens: int,
        final: int = 5,
    ) -> AssembledContext:
        target = query.target
        context_window = target.context_window
        output_reserve = target.output_reserve_tokens or int(context_window * 0.25)
        query_tokens = self.tokenizer.count(query.text)
        safety_margin = int(context_window * self.config.safety_margin_pct)
        memory_budget = (
            context_window
            - output_reserve
            - query_tokens
            - template_overhead_tokens
            - safety_margin
        )

        if memory_budget <= 0:
            components = {
                "output_reserve": output_reserve,
                "query_tokens": query_tokens,
                "template_overhead": template_overhead_tokens,
                "safety_margin": safety_margin,
            }
            largest = max(components, key=lambda k: components[k])
            slog.warning(
                event="budget_invalid",
                trace_id=query.trace_id,
                context_window=context_window,
                memory_budget=memory_budget,
                largest_component=largest,
            )
            raise BudgetExceededError(
                f"memory_budget={memory_budget} <= 0; largest reservation: "
                f"{largest}={components[largest]}; context_window={context_window}"
            )

        # Reason-from-scratch path: produce an empty context with the flag.
        if resolved.reason_from_scratch:
            return AssembledContext(
                selected=[],
                query=query,
                budget_total=memory_budget,
                budget_remaining=memory_budget,
                dropped_ids=[],
                metadata={
                    "reason_from_scratch": True,
                    "notes": list(resolved.notes),
                    "reasoner_thinking": resolved.reasoner_thinking,
                },
                improved_query=resolved.improved_query,
                reasoning_content=resolved.reasoning_content,
            )

        if not resolved.kept:
            return AssembledContext(
                selected=[],
                query=query,
                budget_total=memory_budget,
                budget_remaining=memory_budget,
                dropped_ids=[],
                metadata={
                    "reason_from_scratch": False,
                    "superseded": list(resolved.superseded),
                    "notes": list(resolved.notes),
                    "reasoner_thinking": resolved.reasoner_thinking,
                },
                improved_query=resolved.improved_query,
                reasoning_content=resolved.reasoning_content,
            )

        # Select candidates: top-`final` by score, plus protected (pinned + recent-N)
        ranked_sorted = sorted(resolved.kept, key=lambda r: r.final_score, reverse=True)
        top_n = ranked_sorted[:final]
        top_n_ids = {r.record.id for r in top_n}

        pinned = [r for r in resolved.kept if r.record.pinned]
        recent_n = sorted(
            resolved.kept, key=lambda r: r.record.timestamp, reverse=True
        )[: self.config.never_drop_recent_n]
        protected_ids = {r.record.id for r in pinned} | {r.record.id for r in recent_n}

        # Candidate pool = union of top_n and protected, preserving final-score order
        candidate_ids = top_n_ids | protected_ids
        candidates = [r for r in ranked_sorted if r.record.id in candidate_ids]

        remaining = memory_budget
        selected: list[RankedHit] = []
        dropped: list[str] = []

        # Protected items first (try to keep them).
        for rh in [c for c in candidates if c.record.id in protected_ids]:
            placed, used, mode = self._try_place(rh, remaining)
            if placed is None:
                dropped.append(rh.record.id)
                continue
            selected.append(placed)
            remaining -= used

        # Then the rest of the top-N (skip already-placed protected).
        placed_ids = {r.record.id for r in selected}
        for rh in [c for c in candidates if c.record.id not in placed_ids]:
            placed, used, mode = self._try_place(rh, remaining)
            if placed is None:
                dropped.append(rh.record.id)
                continue
            selected.append(placed)
            remaining -= used

        selected.sort(key=lambda r: r.final_score, reverse=True)
        # Strict cap at `final` — protected items can push the count if they
        # were not in top_n. Trim to `final` by score, but never drop pinned.
        if len(selected) > final:
            pinned_picks = [r for r in selected if r.record.pinned]
            unpinned_picks = [r for r in selected if not r.record.pinned]
            keep_unpinned = unpinned_picks[: max(0, final - len(pinned_picks))]
            for r in unpinned_picks[len(keep_unpinned):]:
                dropped.append(r.record.id)
            selected = sorted(pinned_picks + keep_unpinned, key=lambda r: r.final_score, reverse=True)

        return AssembledContext(
            selected=selected,
            query=query,
            budget_total=memory_budget,
            budget_remaining=remaining,
            dropped_ids=dropped,
            metadata={
                "reason_from_scratch": False,
                "superseded": list(resolved.superseded),
                "notes": list(resolved.notes),
                "reasoner_thinking": resolved.reasoner_thinking,
                "cache_hits": self.cache_hits,
                "precomputed_hits": self.precomputed_hits,
                "cache_misses": self.cache_misses,
            },
            improved_query=resolved.improved_query,
            reasoning_content=resolved.reasoning_content,
        )

    def _try_place(
        self, rh: RankedHit, remaining: int
    ) -> tuple[RankedHit | None, int, str]:
        """Try to fit ``rh`` at full → medium → short. Returns (placed, tokens, mode)
        or (None, 0, "dropped")."""
        item = rh.record
        full = self._count_full(item)
        if full <= remaining:
            return rh, full, "full"
        if item.summary_medium is not None:
            med = self._count_medium(item)
            if med <= remaining:
                new_record = item.model_copy(update={"content": item.summary_medium})
                new_hit = rh.hit.model_copy(update={"record": new_record})
                return rh.model_copy(update={"hit": new_hit}), med, "medium"
        if item.summary_short is not None:
            short = self._count_short(item)
            if short <= remaining:
                new_record = item.model_copy(update={"content": item.summary_short})
                new_hit = rh.hit.model_copy(update={"record": new_record})
                return rh.model_copy(update={"hit": new_hit}), short, "short"
        return None, 0, "dropped"
