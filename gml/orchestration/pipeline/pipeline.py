"""Pipeline — top-level orchestrator.

This is the only place the "found anything?" decision lives. Stage modules
have no knowledge of each other; the branch and the two-site SAM invocation
are explicit here.

Flow:

    user query (plain text)
        → Classifier        (classify query intent/type)
        → Embedder          (text → vector)
        → Retriever.search  (vector DB search)
        → branch on "found anything?":
            NO  → SAM.reason_from_scratch()      (no memory context)
            YES → Retriever.get_top_matches(50)
                → Reranker.pick_best(10)
                → SAM.resolve_conflicts()        (old vs new memories)
        → Assembler.package(final=5)
        → Translator        (format payload for target AI)
"""
import asyncio
import hashlib
import os
import re
import time
import uuid
from typing import Callable

from orchestration.assembler.budget_assembler import BudgetAssembler
from orchestration.classifier.base import Classifier
from orchestration.embedder.base import Embedder
from orchestration.errors import BudgetExceededError, TranslatorError
from orchestration.observability.logging import StructuredLogger
from orchestration.observability.metrics import (
    ORCHESTRATION_BUDGET_UTILIZATION_RATIO,
    ORCHESTRATION_FAILURES_TOTAL,
    ORCHESTRATION_LATENCY_SECONDS,
    ORCHESTRATION_REQUESTS_TOTAL,
    ORCHESTRATION_RETRIEVAL_LATENCY_SECONDS,
)
from orchestration.pipeline.contracts import (
    AssembledContext,
    Classification,
    ClassificationSource,
    ModelFamily,
    OrchestrationConfig,
    Query,
    RankedHit,
    ResolvedMemorySet,
    TargetDescriptor,
    TranslatedPayload,
)
from orchestration.reranker.base import Reranker
from orchestration.retriever.base import Retriever
from orchestration.sam.sam import SAM
from orchestration.sdp.query_router import classify_query
from orchestration.tokenizers.base import Tokenizer
from orchestration.tokenizers.claude_tokenizer import ClaudeTokenizer
from orchestration.tokenizers.deepseek_tokenizer import DeepSeekTokenizer
from orchestration.tokenizers.gemini_tokenizer import GeminiTokenizer
from orchestration.tokenizers.llama_tokenizer import LlamaTokenizer
from orchestration.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from orchestration.translator.translator import Translator


slog = StructuredLogger("pipeline")

_PLEASANTRY_REGEX = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay|cool|nice)[\s!.?]*$",
    re.IGNORECASE,
)


# Thresholds for the SAM-skip heuristic — see should_skip_sam(). Both
# tunable via env vars.
#
# The default rule (raised from 0.75 → 0.85 after LOCOMO measurement
# showed many wrong-answer top hits scoring 0.78-0.82):
#   skip iff
#     top_score >= SAM_SKIP_DEFAULT_THRESHOLD          (.85)
#     AND
#     (top1.score - top2.score) >= SAM_SKIP_DEFAULT_GAP  (.10)
#     AND
#     no two top-N memories share entity+attribute with different value
#
# The gap requirement is what protects against the LOCOMO failure mode:
# multiple competing memories that ALL score >= 0.85 mean ambiguity →
# run SAM and let it improve the query.
# Adjusted for jina-reranker-v2-base sigmoid scale where relevant scores
# cluster around 0.50-0.70 (not 0.95+ like MiniLM). Threshold 0.70 means
# "the cross-encoder is highly confident this is relevant"; gap 0.15 means
# "top-1 is clearly better than top-2". Both overridable via env.
#
# Raised 0.55 -> 0.70: skip SAM only when the top score is genuinely high, so
# the cross-encoder / SAM runs in more ambiguous mid-confidence cases instead
# of short-circuiting them.
SAM_SKIP_DEFAULT_THRESHOLD = 0.70
SAM_SKIP_DEFAULT_GAP = 0.15


def _has_entity_attribute_conflict(ranked: list[RankedHit]) -> bool:
    """True if any two memories share (entity, attribute) but disagree on value.

    This is the only situation where SAM's conflict-resolution adds value:
    same fact, different versions, need to pick a winner. If every memory
    is about a different fact, there's nothing for SAM to resolve.
    """
    seen: dict[tuple[str, str], set] = {}
    for rh in ranked:
        rec = rh.record if hasattr(rh, "record") else rh.hit.record
        if not rec.entity or not rec.attribute:
            continue
        key = (rec.entity, rec.attribute)
        seen.setdefault(key, set()).add(rec.value)
        if len(seen[key]) > 1:
            return True
    return False


def should_skip_sam(
    ranked: list[RankedHit],
    threshold: float | None = None,
    gap: float | None = None,
    query_text: str | None = None,
    hints=None,
) -> tuple[bool, str]:
    """Decide whether SAM.resolve_conflicts can be safely skipped.

    Returns (skip, reason). Skip when ANY of these hold:
      A. Question type guard (Tier-3 fix): multi-hop / temporal / list-style
         questions. Measured on held-out LOCOMO (100 QA):
           cat-2 multi-hop  F1: 0.107 (SAM on) → 0.175 (SAM off)  +0.068
           cat-3 temporal   F1: 0.167 (SAM on) → 0.223 (SAM off)  +0.056
           cat-1 single-hop F1: 0.391 (SAM on) → 0.361 (SAM off)  -0.030
         SAM's "improved_query" narrows the question (e.g.
         "What practices does Caroline do?" → "What specific YOGA practices?")
         which drops list items. Skip SAM where the narrowing hurts.
      B. Reranker is confident: top_score >= threshold AND
         (top1.score - top2.score) >= gap AND no entity+attribute conflict.

    Override with env vars:
      GML_SAM_DISABLED=1               — never run SAM (force skip)
      GML_FORCE_SAM=1                  — always run SAM (disable both skips)
      GML_SAM_CONDITIONAL=0            — disable type-based skip (Tier-3 fix)
      GML_SAM_SKIP_THRESHOLD=<float>   — change the min top-score
      GML_SAM_SKIP_GAP=<float>         — change the min top1-top2 gap
    """
    if os.environ.get("GML_SAM_DISABLED") == "1":
        return True, "GML_SAM_DISABLED=1 forces SAM off"
    if os.environ.get("GML_FORCE_SAM") == "1":
        return False, "GML_FORCE_SAM=1 forces SAM on"
    if not ranked:
        return False, "no ranked hits"

    # (A) Type-based conditional skip — Tier 3 fix. Default ON; set
    # GML_SAM_CONDITIONAL=0 to disable and use confidence-only logic.
    if os.environ.get("GML_SAM_CONDITIONAL", "1") == "1":
        # Prefer caller-supplied hints; otherwise compute from query_text.
        if hints is None and query_text:
            from orchestration.sdp.query_router import classify_query
            hints = classify_query(query_text)
        if hints is not None:
            if hints.is_multi_hop:
                return True, "multi-hop question — SAM narrowing hurts cat-2 F1"
            if hints.is_temporal:
                return True, "temporal question — SAM narrowing hurts cat-3 F1"
        if query_text:
            from orchestration.sam.answer_generator import _is_list_question
            if _is_list_question(query_text):
                return True, "list-style question — SAM rewrite drops list items"

    threshold = threshold if threshold is not None else float(
        os.environ.get("GML_SAM_SKIP_THRESHOLD", SAM_SKIP_DEFAULT_THRESHOLD)
    )
    gap = gap if gap is not None else float(
        os.environ.get("GML_SAM_SKIP_GAP", SAM_SKIP_DEFAULT_GAP)
    )
    top_score = ranked[0].final_score
    if top_score < threshold:
        return False, f"top_score={top_score:.2f} < threshold {threshold:.2f} (low confidence)"
    if len(ranked) >= 2:
        top1_top2 = top_score - ranked[1].final_score
        if top1_top2 < gap:
            return (
                False,
                f"top1-top2 gap {top1_top2:.2f} < {gap:.2f} (ambiguous; need SAM)"
            )
    if _has_entity_attribute_conflict(ranked):
        return False, "entity+attribute conflict in top hits"
    return True, f"unambiguous (top={top_score:.2f}, gap={top_score - (ranked[1].final_score if len(ranked) >= 2 else 0):.2f})"


TokenizerFactory = Callable[[TargetDescriptor], Tokenizer]


def default_tokenizer_factory(target: TargetDescriptor) -> Tokenizer:
    """Dispatch tokenizer by target family. Used when no factory is supplied."""
    if target.model_family == ModelFamily.GPT:
        return TiktokenTokenizer(target.model_version)
    if target.model_family == ModelFamily.GEMINI:
        return GeminiTokenizer()
    if target.model_family == ModelFamily.CLAUDE:
        return ClaudeTokenizer()
    if target.model_family == ModelFamily.LLAMA:
        return LlamaTokenizer()
    if target.model_family == ModelFamily.DEEPSEEK:
        return DeepSeekTokenizer()
    if target.model_family == ModelFamily.CURSOR:
        backend = target.cursor_backend or target.model_version
        return TiktokenTokenizer(backend)
    raise NotImplementedError(
        f"No default tokenizer for model_family={target.model_family!r}"
    )


class Pipeline:
    """End-to-end pipeline: plain-text user query → TranslatedPayload.

    The seven stages are passed in. Cross-stage flow lives in :meth:`run`.

    Example:
        >>> pipeline = Pipeline(
        ...     classifier=KeywordClassifier(),
        ...     embedder=StubEmbedder(),
        ...     retriever=StubRetriever(),
        ...     reranker=ScoreReranker(config),
        ...     sam=SAM(),
        ...     translator=Translator(),
        ...     config=config,
        ... )
        >>> payload = await pipeline.run(Pipeline.build_query(
        ...     "how do I fix the auth bug?",
        ...     target=TargetDescriptor.for_claude(),
        ... ))
    """

    def __init__(
        self,
        *,
        classifier: Classifier,
        embedder: Embedder,
        retriever: Retriever,
        reranker: Reranker,
        sam: SAM,
        translator: Translator,
        config: OrchestrationConfig,
        tokenizer_factory: TokenizerFactory | None = None,
    ) -> None:
        self.classifier = classifier
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.sam = sam
        self.translator = translator
        self.config = config
        self._tokenizer_factory: TokenizerFactory = (
            tokenizer_factory or default_tokenizer_factory
        )
        self._config_hash = hashlib.sha256(
            config.model_dump_json().encode("utf-8")
        ).hexdigest()[:12]
        # Memoize the (target_family, model_version) → (tokenizer, assembler,
        # template_overhead_tokens) triple so per-target setup is paid once.
        self._target_cache: dict[tuple[ModelFamily, str], tuple[Tokenizer, BudgetAssembler, int]] = {}

    # ------------------------------------------------------------------
    # Convenience constructor for callers that just want to pass text.
    # ------------------------------------------------------------------

    @staticmethod
    def build_query(
        text: str,
        target: TargetDescriptor,
        session_context: dict | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        as_of=None,
    ) -> Query:
        if as_of is None:
            # NL time-travel: "what was our stack in March 2025?" resolves
            # to a concrete as_of so temporal retrieval works
            # conversationally, not just via the HTTP parameter. An
            # explicitly supplied as_of always wins. GML_NL_AS_OF=0 disables.
            from orchestration.sdp.temporal_parser import parse_as_of
            as_of = parse_as_of(text)
        return Query(
            text=text,
            target=target,
            session_context=session_context or {},
            trace_id=trace_id or uuid.uuid4().hex,
            user_id=user_id,
            as_of=as_of,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: Query) -> TranslatedPayload:
        target_family = query.target.model_family.value
        ORCHESTRATION_REQUESTS_TOTAL.inc(target_family=target_family)
        slog.info(
            event="pipeline_start",
            trace_id=query.trace_id,
            target_family=target_family,
            query_length=len(query.text),
        )
        t_start = time.perf_counter()

        try:
            tokenizer, assembler, template_overhead = self._resolve_target(query.target)

            # ---- Pleasantry short-circuit ---------------------------------
            # Spec puts the YES/NO branch in the pipeline; pleasantries are
            # the same kind of flow decision, so they live here too.
            if _PLEASANTRY_REGEX.match(query.text):
                slog.info(event="pleasantry_short_circuit", trace_id=query.trace_id)
                classification = Classification(
                    intent_type="pleasantry",
                    entities=[],
                    retrieval_hints={},
                    confidence=1.0,
                    source=ClassificationSource.FAST_PATH,
                )
                resolved = await self.sam.reason_from_scratch(query, classification)
                context = assembler.package(
                    resolved, query, template_overhead_tokens=template_overhead,
                    final=self.config.assembler_final_k,
                )
                context = self._mark_pleasantry(context)
                return self.translator.translate(context, config_hash=self._config_hash)

            # ---- 1. Classifier --------------------------------------------
            classification = await self._stage_classifier(query)

            # ---- 2. Embedder ----------------------------------------------
            embedded = await self.embedder.embed(query, classification)

            # ---- 3. Retriever.search (probe) ------------------------------
            t_ret = time.perf_counter()
            try:
                timeout_s = self.config.timeouts_per_stage_ms["retriever"] / 1000.0
                hits = await asyncio.wait_for(
                    self.retriever.search(embedded), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                ORCHESTRATION_FAILURES_TOTAL.inc(stage="retriever")
                slog.warning(
                    event="retriever_timeout",
                    trace_id=query.trace_id,
                    timeout_seconds=timeout_s,
                    degraded_mode=True,
                )
                hits = []
            except Exception as exc:
                ORCHESTRATION_FAILURES_TOTAL.inc(stage="retriever")
                slog.warning(
                    event="retriever_failed",
                    trace_id=query.trace_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    degraded_mode=True,
                )
                hits = []
            finally:
                ORCHESTRATION_RETRIEVAL_LATENCY_SECONDS.observe(
                    time.perf_counter() - t_ret
                )

            # ---- 4. "found anything?" branch ------------------------------
            if not hits:
                slog.info(
                    event="branch_no_match",
                    trace_id=query.trace_id,
                    hits=0,
                )
                resolved = await self.sam.reason_from_scratch(query, classification)
            else:
                # C1: query router — temporal/multi-hop/count questions
                # need a wider candidate pool to give the reranker enough
                # to work with. Pure heuristic, no LLM call.
                hints = classify_query(query.text)
                adjusted_top_k = int(
                    self.config.retriever_top_k * hints.top_k_multiplier
                )
                slog.info(
                    event="branch_yes_match",
                    trace_id=query.trace_id,
                    initial_hits=len(hits),
                    query_hints=hints.notes,
                    top_k_multiplier=hints.top_k_multiplier,
                    adjusted_top_k=adjusted_top_k,
                )
                top50 = await self.retriever.get_top_matches(
                    embedded, k=adjusted_top_k
                )
                # Graph-aware expansion: pull in the 1-hop semantic
                # neighbours of the strongest seeds so related facts the
                # query embedding missed still reach the reranker. The
                # reranker stays the quality gate for what survives.
                top50 = await self._expand_neighbors(embedded, top50)
                top10 = await self.reranker.pick_best(
                    top50, query, k=self.config.reranker_top_k
                )
                skip, reason = should_skip_sam(
                    top10, query_text=query.text, hints=hints,
                )
                if skip:
                    slog.info(
                        event="sam_skipped",
                        trace_id=query.trace_id,
                        reason=reason,
                        top_score=top10[0].final_score if top10 else 0.0,
                    )
                    resolved = ResolvedMemorySet(
                        kept=top10,
                        superseded=[],
                        reason_from_scratch=False,
                        notes=[f"SAM skipped: {reason}"],
                    )
                else:
                    resolved = await self.sam.resolve_conflicts(query, top10)

                # ---- Iterative + Self-RAG retrieval -----------------------
                # Trigger a second retrieval pass if EITHER:
                #   1. SAM produced an improved_query that differs from the
                #      original (original audit bug fix), OR
                #   2. classify_query says is_multi_hop (Self-RAG: augment
                #      the query with top-1's content/entities for a
                #      multi-hop-aware second pass).
                #
                # Guarded by max-depth=1 so we never recurse. Skip entirely
                # if env GML_ITERATIVE_RETRIEVAL=0.
                want_iterative = os.environ.get("GML_ITERATIVE_RETRIEVAL", "1") == "1"
                sam_rewrote = bool(
                    resolved.improved_query
                    and resolved.improved_query.strip()
                    and resolved.improved_query.strip() != query.text.strip()
                )
                # Self-RAG fires when:
                #   - query is multi-hop (multi-step reasoning needed), OR
                #   - top reranker score is low (likely paraphrase mismatch)
                # The low-score trigger catches cat-1 single-hop paraphrastic
                # questions ("what did X research?" vs evidence "X was looking
                # into Y") that don't lexically match the embedded form.
                self_rag_low_score_threshold = float(
                    os.environ.get("GML_SELF_RAG_LOW_SCORE", "0.45")
                )
                low_top_score = bool(
                    top10 and top10[0].final_score < self_rag_low_score_threshold
                )
                want_self_rag = (
                    os.environ.get("GML_SELF_RAG", "1") == "1"
                    and top10
                    and (hints.is_multi_hop or low_top_score)
                )

                # ── Tier 3.3: Query decomposition (multi-hop only) ─────
                # When the query is multi-hop AND the decomposer can split
                # it into sub-questions (e.g. "What X and Y does Z do?" →
                # 2 sub-queries), retrieve once per sub-q and merge. Wider
                # candidate pool catches multi-fact answers where a single
                # embedding biases toward one side of the conjunction.
                # Set GML_QUERY_DECOMPOSE=0 to disable.
                decompose_fired = False
                if (
                    want_iterative
                    and hints.is_multi_hop
                    and os.environ.get("GML_QUERY_DECOMPOSE", "1") == "1"
                ):
                    from orchestration.sdp.query_decomposer import decompose
                    sub_queries = decompose(query.text)
                    # decompose() always includes the original as [0]; real
                    # decomposition is signalled by len > 1.
                    extra_sqs = sub_queries[1:3]  # cap at 2 extras (latency)
                    if extra_sqs:
                        slog.info(
                            event="query_decompose_start",
                            trace_id=query.trace_id,
                            n_sub=len(extra_sqs),
                            sub_queries=[s[:80] for s in extra_sqs],
                        )
                        sub_hits_all: list = []
                        for sq_text in extra_sqs:
                            try:
                                sq = query.model_copy(update={"text": sq_text})
                                sq_embedded = await self.embedder.embed(sq, classification)
                                sq_hits = await self.retriever.get_top_matches(
                                    sq_embedded, k=adjusted_top_k
                                )
                                sub_hits_all.extend(sq_hits)
                            except Exception as exc:
                                slog.warning(
                                    event="query_decompose_subq_failed",
                                    trace_id=query.trace_id,
                                    error_type=type(exc).__name__,
                                )
                        if sub_hits_all:
                            seen_ids = {h.record.id for h in top50}
                            new_hits: list = []
                            for h in sub_hits_all:
                                if h.record.id not in seen_ids:
                                    new_hits.append(h)
                                    seen_ids.add(h.record.id)
                            merged = list(top50) + new_hits
                            slog.info(
                                event="query_decompose_merged",
                                trace_id=query.trace_id,
                                original_n=len(top50),
                                new_n=len(new_hits),
                                merged_total=len(merged),
                            )
                            reranked = await self.reranker.pick_best(
                                merged, query, k=self.config.reranker_top_k
                            )
                            resolved = resolved.model_copy(update={
                                "kept": reranked,
                                "notes": [
                                    f"decompose: +{len(new_hits)} new candidates "
                                    f"from {len(extra_sqs)} sub-queries",
                                    *(resolved.notes or []),
                                ],
                            })
                            decompose_fired = True

                # When decomposition handled the multi-hop case, skip the
                # Self-RAG single-augmentation pass (would duplicate work).
                if want_iterative and not decompose_fired and (sam_rewrote or want_self_rag):
                    # Pick the augmented query: SAM's rewrite wins if present,
                    # otherwise build one from top-1's content+entities (Self-RAG).
                    if sam_rewrote:
                        iq_text = resolved.improved_query.strip()
                        iq_source = "sam_rewrite"
                    else:
                        top1 = top10[0].hit.record
                        # Seed with original query + a short snippet of top-1 +
                        # any entities found in top-1's content. Keeps embedding
                        # focused on the question while adding multi-hop context.
                        snippet = top1.content[:160]
                        entities_snippet = ""
                        if top1.entity:
                            entities_snippet = f" Entities: {top1.entity}"
                            if top1.attribute:
                                entities_snippet += f"/{top1.attribute}"
                        iq_text = f"{query.text} (related: {snippet}{entities_snippet})"
                        iq_source = "self_rag_top1_seed"
                    slog.info(
                        event="iterative_retrieval_start",
                        trace_id=query.trace_id,
                        original_query=query.text[:120],
                        improved_query=iq_text[:120],
                        source=iq_source,
                    )
                    iq = query.model_copy(update={"text": iq_text})
                    try:
                        iq_embedded = await self.embedder.embed(iq, classification)
                        iq_hits = await self.retriever.get_top_matches(
                            iq_embedded, k=adjusted_top_k
                        )
                    except Exception as exc:
                        slog.warning(
                            event="iterative_retrieval_failed",
                            trace_id=query.trace_id,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        iq_hits = []

                    # Take the union of the original top50 + iterative hits,
                    # deduped by record id, then rerank fresh.
                    #
                    # Importantly: do NOT call SAM a second time. SAM has
                    # already produced improved_query + reasoning_content
                    # for THIS turn; calling it again would either burn
                    # another LLM call OR (when the test mock runs out)
                    # silently overwrite the good outputs with empty
                    # heuristic ones. We just update the kept set.
                    if iq_hits:
                        seen_ids = {h.record.id for h in top50}
                        merged = list(top50) + [
                            h for h in iq_hits if h.record.id not in seen_ids
                        ]
                        slog.info(
                            event="iterative_retrieval_merged",
                            trace_id=query.trace_id,
                            original_n=len(top50),
                            new_n=len(merged) - len(top50),
                            merged_total=len(merged),
                        )
                        reranked = await self.reranker.pick_best(
                            merged, iq, k=self.config.reranker_top_k
                        )
                        # Preserve resolved.improved_query / .reasoning_content
                        # from the first SAM pass; only update the kept set.
                        resolved = resolved.model_copy(update={
                            "kept": reranked,
                            "notes": [
                                f"iterative_retrieval: +{len(merged) - len(top50)} new candidates",
                                *(resolved.notes or []),
                            ],
                        })

            # ---- 5. Assembler ---------------------------------------------
            context = assembler.package(
                resolved,
                query,
                template_overhead_tokens=template_overhead,
                final=self.config.assembler_final_k,
            )

            # ---- 6. Translator --------------------------------------------
            payload = self.translator.translate(context, config_hash=self._config_hash)

            # Bench-debug hook: expose the FINAL kept hits' source session IDs
            # so the LOCOMO bench can compute retrieval-stage recall@K against
            # gold ``evidence_session``. Off by default — set GML_BENCH_TRACE_HITS=1.
            if os.environ.get("GML_BENCH_TRACE_HITS", "0") == "1":
                try:
                    sess_ids: list = []
                    seen_ids: set = set()
                    for rh in (getattr(resolved, "kept", None) or []):
                        rm = rh.hit.record.raw_metadata or {}
                        sid = rm.get("session_id")
                        if sid is not None and sid not in seen_ids:
                            sess_ids.append(sid)
                            seen_ids.add(sid)
                    payload.metadata["top_session_ids"] = sess_ids
                except Exception:
                    pass

            return payload

        except (BudgetExceededError, TranslatorError) as exc:
            ORCHESTRATION_FAILURES_TOTAL.inc(stage=type(exc).__name__)
            slog.error(
                event="pipeline_failed",
                trace_id=query.trace_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            elapsed = time.perf_counter() - t_start
            ORCHESTRATION_LATENCY_SECONDS.observe(
                elapsed, target_family=target_family
            )
            slog.info(
                event="pipeline_complete",
                trace_id=query.trace_id,
                duration_ms=int(elapsed * 1000),
            )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    async def _expand_neighbors(self, embedded, hits):
        """1-hop graph expansion (GML_GRAPH_EXPANSION=0 disables).

        For the top GML_GRAPH_SEEDS hits, fetch each seed's
        GML_GRAPH_NEIGHBORS_K nearest active memories from the retriever
        and merge unseen ones into the candidate pool. Resolves
        ``get_neighbors`` through wrapper stacks (TimeAware/EntityBoosted
        ``.base``, Hybrid ``.dense``); silently no-ops for retrievers
        without neighbor support. Failures never break retrieval.
        """
        if not hits or os.environ.get("GML_GRAPH_EXPANSION", "1") != "1":
            return hits
        source = self.retriever
        for _ in range(4):  # unwrap nested wrappers, bounded
            if hasattr(source, "get_neighbors"):
                break
            inner = getattr(source, "base", None) or getattr(source, "dense", None)
            if inner is None:
                return hits
            source = inner
        if not hasattr(source, "get_neighbors"):
            return hits

        seeds = hits[: int(os.environ.get("GML_GRAPH_SEEDS", "3"))]
        k = int(os.environ.get("GML_GRAPH_NEIGHBORS_K", "3"))
        results = await asyncio.gather(
            *[source.get_neighbors(embedded, s.record.id, k=k) for s in seeds],
            return_exceptions=True,
        )
        seen = {h.record.id for h in hits}
        out = list(hits)
        added = 0
        for res in results:
            if isinstance(res, BaseException):
                continue
            for h in res:
                if h.record.id not in seen:
                    seen.add(h.record.id)
                    out.append(h)
                    added += 1
        if added:
            slog.info(
                event="graph_expansion",
                trace_id=embedded.query.trace_id,
                seeds=len(seeds),
                added=added,
                pool=len(out),
            )
        return out

    async def _stage_classifier(self, query: Query) -> Classification:
        try:
            # Enforce the configured stage budget here so it applies to any
            # Classifier implementation, not just ones with internal timeouts.
            timeout_s = self.config.timeouts_per_stage_ms["classifier"] / 1000.0
            classification = await asyncio.wait_for(
                self.classifier.classify(query), timeout=timeout_s
            )
            if classification.degraded:
                ORCHESTRATION_FAILURES_TOTAL.inc(stage="classifier")
            return classification
        except Exception as exc:
            ORCHESTRATION_FAILURES_TOTAL.inc(stage="classifier")
            slog.warning(
                event="classifier_failed_default_intent",
                trace_id=query.trace_id,
                error_type=type(exc).__name__,
                error=str(exc),
                degraded_mode=True,
            )
            return Classification(
                intent_type="other",
                entities=[],
                retrieval_hints={},
                confidence=0.0,
                source=ClassificationSource.KEYWORD_FALLBACK,
                degraded=True,
            )

    # ------------------------------------------------------------------
    # Target-specific cache
    # ------------------------------------------------------------------

    def _resolve_target(
        self, target: TargetDescriptor
    ) -> tuple[Tokenizer, BudgetAssembler, int]:
        key = (target.model_family, target.model_version)
        cached = self._target_cache.get(key)
        if cached is not None:
            return cached
        tokenizer = self._tokenizer_factory(target)
        assembler = BudgetAssembler(tokenizer, self.config)
        adapter = self.translator.adapter_for(target)
        overhead = tokenizer.count(adapter.empty_template())
        triple = (tokenizer, assembler, overhead)
        self._target_cache[key] = triple
        return triple

    @staticmethod
    def _mark_pleasantry(context: AssembledContext) -> AssembledContext:
        meta = dict(context.metadata)
        meta["short_circuit"] = "pleasantry"
        return context.model_copy(update={"metadata": meta})
