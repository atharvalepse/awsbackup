"""Verbose pipeline trace — full text of every stage on a single query.

Builds a 2-session synthetic conversation, ingests it, then walks the
query through every pipeline stage and DUMPS the actual content each
stage produced — every retrieval hit in full, the LLM prompts SAM and
the answer generator sent, the raw JSON SAM got back, the entire
assembled context, the full translator payload, and the raw model
output before post-processing.

Use this when you want to understand WHY a query produced a particular
answer, or to debug a specific stage. Output is long — pipe through
``less -R`` for color.

Run:
    .venv/bin/python scripts/trace_pipeline.py
    .venv/bin/python scripts/trace_pipeline.py --query "What car does Caroline drive?" --category 1
    .venv/bin/python scripts/trace_pipeline.py --no-answer-gen   # skip 5s LLM call
"""
import argparse
import asyncio
import json
import os
import sys
import textwrap
import time

sys.path.insert(0, ".")

from orchestration.pipeline import Pipeline, TargetDescriptor
from orchestration.pipeline.contracts import ResolvedMemorySet
from orchestration.pipeline.pipeline import should_skip_sam
from orchestration.sam._ollama_client import OllamaClient, make_local_llm_client
from orchestration.sam.answer_generator import AnswerGenerator
from orchestration.sdp.query_router import classify_query
from scripts.benchmark_locomo import _build_pipeline_with_temp_store, ingest_session_raw


SESSION_1 = {
    "session_id": 1,
    "date": "2026-04-12T10:00:00",
    "messages": [
        {"speaker": "Caroline", "content": "Hey Mel, how have you been?"},
        {"speaker": "Mel",      "content": "Good. Just got back from Tokyo."},
        {"speaker": "Caroline", "content": "I went to a yoga class last Thursday — it was great."},
        {"speaker": "Mel",      "content": "What style?"},
        {"speaker": "Caroline", "content": "Hot yoga. The teacher is named Priya."},
        {"speaker": "Caroline", "content": "I drive a blue Prius these days."},
    ],
}
SESSION_2 = {
    "session_id": 2,
    "date": "2026-05-03T10:00:00",
    "messages": [
        {"speaker": "Caroline", "content": "I started a meditation class on Monday."},
        {"speaker": "Mel",      "content": "How's it going?"},
        {"speaker": "Caroline", "content": "Amazing. I no longer feel anxious before work."},
        {"speaker": "Mel",      "content": "Are you still doing yoga?"},
        {"speaker": "Caroline", "content": "Yes — yoga Thursdays, meditation Mondays."},
    ],
}

# ANSI
B = "\033[1m"; D = "\033[2m"; CY = "\033[36m"; YE = "\033[33m"
GR = "\033[32m"; MG = "\033[35m"; RD = "\033[31m"; R = "\033[0m"


def stage(num: int, name: str, dur_ms: float | None = None) -> None:
    bar = "═" * 72
    tag = f"  {D}[{dur_ms:.0f}ms]{R}" if dur_ms is not None else ""
    print(f"\n{B}{CY}{bar}{R}")
    print(f"{B}{CY}║ Stage {num}: {name}{R}{tag}")
    print(f"{B}{CY}{bar}{R}")


def sub(label: str, indent: int = 2) -> None:
    print(f"{' ' * indent}{B}{D}── {label} ──{R}")


def kv(key: str, value, indent: int = 2) -> None:
    print(f"{' ' * indent}{D}{key}:{R} {value}")


def wrap(text: str, indent: int = 4, width: int = 96) -> None:
    if not text:
        print(f"{' ' * indent}{D}(empty){R}")
        return
    for line in text.splitlines() or [""]:
        for sub_line in textwrap.wrap(line, width=width) or [""]:
            print(f"{' ' * indent}{sub_line}")


def dump_block(text: str, indent: int = 4, max_lines: int | None = None) -> None:
    """Print text as-is, indented. If max_lines is set, cap and note overflow."""
    if not text:
        print(f"{' ' * indent}{D}(empty){R}")
        return
    lines = text.splitlines()
    shown = lines if max_lines is None else lines[:max_lines]
    for line in shown:
        print(f"{' ' * indent}{line}")
    if max_lines is not None and len(lines) > max_lines:
        print(f"{' ' * indent}{D}… +{len(lines) - max_lines} more lines{R}")


class CapturingClient(OllamaClient):
    """Wraps an OllamaClient and records every prompt+response pair."""

    def __init__(self, inner: OllamaClient) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    async def generate(self, prompt, *, json_mode=False, max_tokens=None):
        gen = await self.inner.generate(
            prompt, json_mode=json_mode, max_tokens=max_tokens
        )
        self.calls.append({
            "prompt": prompt,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
            "thinking": gen.thinking,
            "answer": gen.answer,
        })
        return gen


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", default="What practices does Caroline do regularly?")
    p.add_argument("--category", type=int, default=2)
    p.add_argument("--no-answer-gen", action="store_true")
    p.add_argument("--no-sam", action="store_true",
                   help="Skip SAM entirely: no LLM reasoning, no conflict "
                        "resolution. Reranked top-K passes straight to "
                        "assembler. Also disables Self-RAG triggered by "
                        "SAM rewrite (multi-hop / low-score self-RAG still fire).")
    p.add_argument("--max-prompt-lines", type=int, default=60,
                   help="Cap prompt dumps at N lines (full output too long otherwise).")
    args = p.parse_args()

    print(f"{B}═══ GML PIPELINE TRACE — verbose ═══{R}")
    print(f"Query:    {YE}{args.query!r}{R}")
    print(f"Category: {args.category}\n")

    # ─── Setup ─────────────────────────────────────────────────────────
    print(f"{B}─── Setup ───{R}")
    pipeline, retriever, store, tmp_path, _aal, entity_index = (
        _build_pipeline_with_temp_store()
    )
    # Wrap the SAM reasoner's client so we capture its LLM calls.
    sam_capture: CapturingClient | None = None
    if pipeline.sam.reasoner is not None:
        sam_capture = CapturingClient(pipeline.sam.reasoner.client)
        pipeline.sam.reasoner.client = sam_capture

    kv("embedder", pipeline.embedder.version)
    kv("retriever", type(retriever).__name__)
    kv("reranker", type(pipeline.reranker).__name__)
    kv("sam_reasoner", "ON (captured)" if sam_capture else "heuristic-only")

    t = time.perf_counter()
    for sess in (SESSION_1, SESSION_2):
        n = await ingest_session_raw(sess, retriever, store)
        kv(f"session-{sess['session_id']}", f"{n} memories ingested")
    kv("ingest total", f"{(time.perf_counter()-t)*1000:.0f}ms · "
       f"{entity_index.entity_count} entities")
    sub("Top entities indexed:")
    for ent, count in entity_index.top_entities(n=10):
        print(f"     {ent}: {count}")

    target = TargetDescriptor.for_claude()
    query = Pipeline.build_query(args.query, target=target)
    tokenizer, assembler, template_overhead = pipeline._resolve_target(target)

    # ─── Stage 1: CLASSIFIER ───────────────────────────────────────────
    t = time.perf_counter()
    classification = await pipeline.classifier.classify(query)
    stage(1, "CLASSIFIER", (time.perf_counter() - t) * 1000)
    kv("query.text", repr(query.text))
    kv("target", f"{target.model_family}/{target.model_version}")
    intent = classification.intent_type
    kv("intent_type", intent.value if hasattr(intent, "value") else intent)
    kv("entities", classification.entities or "—")
    kv("retrieval_hints", classification.retrieval_hints or "—")
    kv("confidence", f"{classification.confidence:.2f}")
    kv("source", classification.source)
    kv("degraded", classification.degraded)

    # ─── Stage 2: EMBEDDER ─────────────────────────────────────────────
    t = time.perf_counter()
    embedded = await pipeline.embedder.embed(query, classification)
    stage(2, "EMBEDDER (dense vector)", (time.perf_counter() - t) * 1000)
    vec = embedded.vector
    norm = sum(v * v for v in vec) ** 0.5
    kv("model", embedded.embedder_version)
    kv("dim", len(vec))
    kv("vector norm", f"{norm:.4f}")
    kv("first 16 dims", [round(float(v), 4) for v in vec[:16]])
    kv("last 8 dims",   [round(float(v), 4) for v in vec[-8:]])
    sub("Vector summary stats:")
    abs_vec = [abs(v) for v in vec]
    print(f"     min={min(vec):.4f}  max={max(vec):.4f}  mean={sum(vec)/len(vec):.4f}  "
          f"|max|={max(abs_vec):.4f}  nonzero={sum(1 for v in vec if abs(v)>1e-6)}/{len(vec)}")

    # ─── Stage 3: QUERY ROUTER ─────────────────────────────────────────
    t = time.perf_counter()
    hints = classify_query(query.text)
    adjusted_top_k = int(pipeline.config.retriever_top_k * hints.top_k_multiplier)
    stage(3, "QUERY ROUTER (heuristic hints)", (time.perf_counter() - t) * 1000)
    kv("category", hints.category)
    kv("is_multi_hop", hints.is_multi_hop)
    kv("is_temporal", hints.is_temporal)
    kv("is_count", hints.is_count)
    kv("is_negation", hints.is_negation)
    kv("top_k_multiplier", hints.top_k_multiplier)
    kv("base retriever_top_k", pipeline.config.retriever_top_k)
    kv("adjusted top_k", adjusted_top_k)

    # ─── Stage 4: RETRIEVER (full hit list) ────────────────────────────
    t = time.perf_counter()
    top_k = await retriever.get_top_matches(embedded, k=adjusted_top_k)
    stage(4, f"RETRIEVER.get_top_matches (k={adjusted_top_k})",
          (time.perf_counter() - t) * 1000)
    kv("hits returned", len(top_k))
    kv("retriever pipeline", "MultiHopAware → EntityBoosted → Hybrid (dense + BM25 RRF)")
    sub(f"All {len(top_k)} retrieval hits (id · similarity · source · content):")
    for i, h in enumerate(top_k, 1):
        rec = h.record
        print(f"  [{i:>2}] {GR}sim={h.similarity:.3f}{R}  "
              f"{D}{rec.id[:18]:18s}{R}  {D}[{rec.source}]{R}")
        wrap(rec.content, indent=8)
        if rec.entity:
            print(f"        {D}entity={rec.entity}/{rec.attribute or '-'} "
                  f"auth={rec.authority_score:.2f} ts={rec.timestamp.date()}{R}")

    # ─── Stage 5: RERANKER ─────────────────────────────────────────────
    t = time.perf_counter()
    reranked = await pipeline.reranker.pick_best(
        top_k, query, k=pipeline.config.reranker_top_k
    )
    stage(5, "RERANKER (CE FT'd + NegationAware wrapper)",
          (time.perf_counter() - t) * 1000)
    kv("hits returned", len(reranked))
    sub(f"All {len(reranked)} reranked hits (full content + per-dim scores):")
    for i, rh in enumerate(reranked, 1):
        rec = rh.hit.record
        print(f"  [{i:>2}] {GR}final={rh.final_score:.3f}{R}  "
              f"sem={rh.semantic_score:.3f}  rec={rh.recency_score:.3f}  "
              f"auth={rh.authority_score:.3f}  pin={rh.pin_boost:.2f}")
        print(f"       {D}id={rec.id} ts={rec.timestamp.date()}{R}")
        wrap(rec.content, indent=8)
        if rh.score_reason:
            print(f"       {D}why: {rh.score_reason}{R}")

    # ─── Stage 6: SAM-skip + resolve_conflicts ─────────────────────────
    t = time.perf_counter()
    if args.no_sam:
        skip, reason = True, "FORCED OFF via --no-sam"
    else:
        skip, reason = should_skip_sam(reranked)
    stage(6, "SAM (skip-or-reason + conflict-resolver)",
          (time.perf_counter() - t) * 1000)
    kv("should_skip_sam()", f"{skip}  {D}({reason}){R}")
    if skip:
        sub("SKIPPED — using reranked as-is")
        resolved = ResolvedMemorySet(
            kept=reranked, superseded=[], reason_from_scratch=False,
            notes=[f"SAM skipped: {reason}"],
        )
    else:
        sub("Calling SAM.resolve_conflicts() → LLMReasoner...")
        t_sam = time.perf_counter()
        try:
            resolved = await pipeline.sam.resolve_conflicts(query, reranked)
            kv("resolve_conflicts duration", f"{(time.perf_counter()-t_sam)*1000:.0f}ms")
        except Exception as exc:
            kv("ERROR", f"{type(exc).__name__}: {exc}")
            resolved = ResolvedMemorySet(
                kept=reranked, superseded=[], reason_from_scratch=False,
                notes=[f"sam failed: {exc}"],
            )

        # Dump SAM's raw LLM I/O if we captured it
        if sam_capture and sam_capture.calls:
            sub("SAM LLM call(s):")
            for ci, call in enumerate(sam_capture.calls, 1):
                print(f"\n  {B}── SAM call #{ci}{R}  json_mode={call['json_mode']} "
                      f"max_tokens={call['max_tokens']}")
                print(f"  {D}── prompt sent ──{R}")
                dump_block(call["prompt"], indent=4,
                           max_lines=args.max_prompt_lines)
                print(f"  {D}── raw LLM thinking ──{R}")
                dump_block(call["thinking"] or "(none)", indent=4, max_lines=20)
                print(f"  {D}── raw LLM answer (JSON) ──{R}")
                dump_block(call["answer"], indent=4, max_lines=40)

        sub("ResolvedMemorySet (after safety nets):")
        kv("kept", len(resolved.kept))
        kv("superseded", len(resolved.superseded))
        kv("improved_query", resolved.improved_query or "—")
        kv("reason_from_scratch", resolved.reason_from_scratch)
        kv("notes", resolved.notes or "—")
        if resolved.reasoning_content:
            sub("SAM reasoning_content:")
            wrap(resolved.reasoning_content, indent=4)
        if resolved.superseded:
            sub("Superseded memory ids:")
            for sup in resolved.superseded:
                rec = sup.hit.record
                print(f"     {rec.id}  →  {rec.content[:80]}")

    # ─── Stage 7: Iterative Retrieval / Self-RAG ───────────────────────
    self_rag_thr = float(os.environ.get("GML_SELF_RAG_LOW_SCORE", "0.45"))
    sam_rewrote = bool(
        resolved.improved_query and resolved.improved_query.strip()
        and resolved.improved_query.strip() != query.text.strip()
    )
    low_top = bool(reranked and reranked[0].final_score < self_rag_thr)
    will_fire = (sam_rewrote or hints.is_multi_hop or low_top) and not skip

    stage(7, "ITERATIVE / Self-RAG (second retrieval pass)")
    kv("sam_rewrote_query?", sam_rewrote)
    kv("is_multi_hop?", hints.is_multi_hop)
    kv("top_score_below_thr?", f"{low_top}  (top={reranked[0].final_score:.3f} < {self_rag_thr}?)")
    kv("will fire?", will_fire)

    if will_fire:
        if sam_rewrote:
            iq_text = resolved.improved_query.strip()
            iq_source = "sam_rewrite"
        else:
            top1 = reranked[0].hit.record
            iq_text = f"{query.text} (related: {top1.content[:160]})"
            iq_source = "self_rag_top1_seed"

        sub(f"Augmented query (source={iq_source}):")
        wrap(iq_text, indent=4)

        t = time.perf_counter()
        iq = query.model_copy(update={"text": iq_text})
        iq_embedded = await pipeline.embedder.embed(iq, classification)
        iq_hits = await retriever.get_top_matches(iq_embedded, k=adjusted_top_k)
        kv("\n  2nd-pass dim", len(iq_embedded.vector))
        kv("  2nd-pass hits", len(iq_hits))
        kv("  duration", f"{(time.perf_counter()-t)*1000:.0f}ms")

        original_ids = {rh.hit.record.id for rh in reranked}
        new_ids = [h.record.id for h in iq_hits if h.record.id not in original_ids]
        sub(f"  NEW memories surfaced by 2nd pass ({len(new_ids)}):")
        for hi in iq_hits:
            if hi.record.id in original_ids:
                continue
            print(f"     {GR}sim={hi.similarity:.3f}{R}  "
                  f"{D}{hi.record.id[:18]}{R}")
            wrap(hi.record.content, indent=8)

    # ─── Stage 8: ASSEMBLER ────────────────────────────────────────────
    t = time.perf_counter()
    context = assembler.package(
        resolved, query,
        template_overhead_tokens=template_overhead,
        final=pipeline.config.assembler_final_k,
    )
    stage(8, f"ASSEMBLER (BudgetAssembler, final={pipeline.config.assembler_final_k})",
          (time.perf_counter() - t) * 1000)
    kv("selected count", len(context.selected))
    kv("dropped count", len(context.dropped_ids))
    kv("budget total tokens", context.budget_total)
    kv("budget remaining", context.budget_remaining)
    kv("budget used", context.budget_total - context.budget_remaining)
    kv("template overhead", template_overhead)
    kv("improved_query (carried)", context.improved_query or "—")
    if context.reasoning_content:
        sub("reasoning_content carried into context:")
        wrap(context.reasoning_content, indent=4)
    sub(f"All {len(context.selected)} selected memories (final ordering):")
    for i, rh in enumerate(context.selected, 1):
        rec = rh.hit.record
        print(f"  [{i:>2}] {GR}score={rh.final_score:.3f}{R}  "
              f"{D}{rec.id[:18]} auth={rec.authority_score:.2f}  src={rec.source}{R}")
        wrap(rec.content, indent=8)
    if context.dropped_ids:
        sub(f"Dropped (budget squeeze): {context.dropped_ids}")

    # ─── Stage 9: TRANSLATOR ───────────────────────────────────────────
    t = time.perf_counter()
    payload = pipeline.translator.translate(context, config_hash=pipeline._config_hash)
    stage(9, f"TRANSLATOR (target={target.model_family}/{target.model_version})",
          (time.perf_counter() - t) * 1000)
    kv("payload_version", payload.payload_version)
    kv("orchestrator_version", payload.orchestrator_version)
    kv("config_hash", payload.config_hash[:16] + "…")
    kv("trace_id", payload.trace_id)
    kv("formatted_context bytes", len(payload.formatted_context))
    kv("user_query", repr(payload.user_query))
    kv("metadata keys", list(payload.metadata.keys()) if payload.metadata else [])
    sub("FULL formatted_context (what the target AI actually sees):")
    print(f"{D}{'─' * 96}{R}")
    print(payload.formatted_context)
    print(f"{D}{'─' * 96}{R}")

    # ─── Stage 10: ANSWER GENERATOR ────────────────────────────────────
    if args.no_answer_gen:
        stage(10, "ANSWER GENERATOR — SKIPPED (--no-answer-gen)")
    else:
        try:
            inner = make_local_llm_client()
            cap = CapturingClient(inner)
            gen = AnswerGenerator(client=cap)
            t = time.perf_counter()
            answer = await gen.answer(
                payload.formatted_context, args.query, category=args.category
            )
            stage(10, "ANSWER GENERATOR (qwen2.5:3b)",
                  (time.perf_counter() - t) * 1000)
            kv("requested category", args.category)
            if cap.calls:
                call = cap.calls[-1]
                kv("max_tokens", call["max_tokens"])
                sub("Prompt sent to LLM:")
                print(f"{D}{'─' * 96}{R}")
                dump_block(call["prompt"], indent=0,
                           max_lines=args.max_prompt_lines)
                print(f"{D}{'─' * 96}{R}")
                sub("Raw LLM response (before post-processing):")
                print(f"  thinking: {call['thinking']!r}")
                print(f"  answer:   {call['answer']!r}")
            sub("Final answer (after _post_process):")
            print(f"  {MG}{B}{answer!r}{R}")
        except Exception as exc:
            stage(10, "ANSWER GENERATOR — error")
            kv("error", f"{type(exc).__name__}: {exc}")

    try:
        tmp_path.unlink()
    except OSError:
        pass

    print(f"\n{B}═══ Trace complete ═══{R}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
