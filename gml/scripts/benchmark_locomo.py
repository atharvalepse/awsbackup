"""LOCOMO benchmark — long-conversation memory eval for the GML pipeline.

LOCOMO (Long Conversational Memory) is a benchmark of multi-session
two-speaker conversations with QA pairs in 5 categories:
  1. single-hop      — answer is in one turn of one session
  2. multi-hop       — answer combines info across multiple sessions
  3. temporal        — answer is about timing/order of events
  4. open-domain     — needs world knowledge plus memory
  5. adversarial     — unanswerable; should return "I don't know" / refuse

This script:
  - Loads LOCOMO data (built-in sample OR a JSON file you pass with --data)
  - For each conversation:
      * Spins up a fresh in-memory pipeline (separate store, won't pollute
        your real ~/.gml/memories.jsonl)
      * Ingests every session's turns via SDP (fast) or the LLM extractor
        (slow, opt-in via --llm-ingest)
      * Runs each QA pair through Pipeline.run and scores it
  - Reports per-category scores + overall aggregates

Scoring:
  - context_recall — does the formatted_context contain the key terms
                     from the gold answer? (lightweight bag-of-words)
  - token_f1      — token-level F1 between gold answer and the words
                     surfaced in formatted_context
  - For category 5 (adversarial): score 1 if the context is empty/short
                     (no false memories injected); 0 otherwise.

Run:
    cd /Users/atharvalepse/Projects/gml-orchestration

    # Quickest — runs on the built-in 2-conversation sample (~30s)
    .venv/bin/python scripts/benchmark_locomo.py

    # Full LOCOMO from a downloaded JSON file (much slower)
    .venv/bin/python scripts/benchmark_locomo.py --data locomo.json --limit 5

    # Compare ingest modes
    .venv/bin/python scripts/benchmark_locomo.py --ingest-mode sdp
    .venv/bin/python scripts/benchmark_locomo.py --ingest-mode llm   # 100x slower
    .venv/bin/python scripts/benchmark_locomo.py --ingest-mode both  # both modes back-to-back
"""
import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# ── LOCOMO date-string parser ─────────────────────────────────────────
# LOCOMO sessions tag dates as "1:56 pm on 8 May, 2023" — neither ISO
# nor any locale standard. The previous code fell through fromisoformat
# (which raised), into the except block, into datetime.now() = 2026.
# Result: every memory got timestamped in 2026 and date_resolver
# computed relative dates ("Sunday") against the wrong anchor.
_LOCOMO_DATE_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),\s*(\d{4})",
    re.IGNORECASE,
)
_MONTH_TO_INT = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_locomo_date(s: str) -> datetime | None:
    """Parse strings like '1:56 pm on 8 May, 2023' → tz-aware datetime.

    Returns None when the string doesn't match. Caller falls back to ISO
    parsing (for synthetic / non-LOCOMO data) and then to datetime.now().
    """
    if not isinstance(s, str):
        return None
    m = _LOCOMO_DATE_RE.match(s)
    if not m:
        return None
    hour, minute, ampm, day, month_str, year = m.groups()
    month = _MONTH_TO_INT.get(month_str.lower())
    if month is None:
        return None
    hour = int(hour) % 12
    if ampm.lower() == "pm":
        hour += 12
    try:
        return datetime(int(year), month, int(day), hour, int(minute), tzinfo=timezone.utc)
    except ValueError:
        return None


def _resolve_session_ts(sess_date) -> datetime:
    """Try LOCOMO format → ISO → datetime.now()."""
    if isinstance(sess_date, str):
        ts = _parse_locomo_date(sess_date)
        if ts is not None:
            return ts
        try:
            ts = datetime.fromisoformat(sess_date.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except (ValueError, AttributeError):
            pass
    return datetime.now(timezone.utc)

from orchestration.classifier import KeywordClassifier
from orchestration.embedder import FastEmbedEmbedder, HydeEmbedder, StubEmbedder
from orchestration.ingestion import MemoryExtractor
from orchestration.memory_store import JsonlMemoryStore
from orchestration.observability.logging import set_output_stream as _set_log_stream
from orchestration.pipeline import Pipeline, TargetDescriptor, load_config
from orchestration.reranker import ScoreReranker, make_reranker
from orchestration.retriever import (
    BM25Retriever, EntityBoostedRetriever, HybridRetriever,
    MultiHopAwareRetriever, SemanticRetriever, TimeAwareRetriever,
)
from orchestration.sam import SAM
from orchestration.sam._ollama_client import make_local_llm_client
from orchestration.sam.aal_record import AALRecord
from orchestration.sam.answer_generator import (
    ANSWER_GEN_ENABLED_DEFAULT, AnswerGenerator,
)
from orchestration.sam.turn_compressor import SAMTurnCompressor
from orchestration.sdp import (
    AAL_ENABLED_DEFAULT, AALTupleExtractor, EntityIndex, HYDE_ENABLED_DEFAULT,
    SDPPipeline, SDPWriter,
)
from orchestration.translator import Translator


# ---------------------------------------------------------------------------
# Built-in LOCOMO-shaped sample. Two conversations, 4 QA each, covering
# the 5 LOCOMO categories. Runs in ~30s end-to-end so you can verify the
# bench works before downloading real LOCOMO.
# ---------------------------------------------------------------------------


BUILTIN_SAMPLE: dict = {
    "conversations": [
        {
            "id": "demo-1-payments",
            "speaker_a": "Alex",
            "speaker_b": "Sam",
            "sessions": [
                {
                    "session_id": 1,
                    "date": "2026-04-12",
                    "messages": [
                        {"speaker": "Alex", "content": "Quick context for you: our payments service is on Stripe and the team lead is Priya Iyer."},
                        {"speaker": "Sam", "content": "Got it — Stripe for payments, Priya leads."},
                        {"speaker": "Alex", "content": "We use PostgreSQL 15 for the orders database. It sits on db-orders-prod-1.internal."},
                        {"speaker": "Sam", "content": "Noted — orders DB is PostgreSQL 15 on db-orders-prod-1.internal."},
                    ],
                },
                {
                    "session_id": 2,
                    "date": "2026-04-25",
                    "messages": [
                        {"speaker": "Alex", "content": "Heads up — we finished migrating payments from Stripe to Adyen yesterday."},
                        {"speaker": "Sam", "content": "Understood. Adyen is the new payment provider."},
                        {"speaker": "Alex", "content": "Also upgraded the orders DB to PostgreSQL 16 last weekend."},
                        {"speaker": "Sam", "content": "Got it — orders DB is now PostgreSQL 16."},
                    ],
                },
                {
                    "session_id": 3,
                    "date": "2026-05-05",
                    "messages": [
                        {"speaker": "Alex", "content": "Priya is moving to a new role next quarter, Marco will take over payments."},
                        {"speaker": "Sam", "content": "Noted — Marco will lead payments going forward."},
                    ],
                },
            ],
            "qa": [
                {"question": "What payment provider do we use?", "answer": "Adyen",
                 "category": 1, "evidence_session": 2},
                {"question": "What database engine is the orders DB on?", "answer": "PostgreSQL 16",
                 "category": 1, "evidence_session": 2},
                {"question": "Who currently leads the payments team and what provider do they use?",
                 "answer": "Marco leads payments and we use Adyen", "category": 2,
                 "evidence_session": 3},
                {"question": "Did we use Stripe before Adyen?",
                 "answer": "Yes, Stripe was the previous provider and we migrated to Adyen",
                 "category": 3, "evidence_session": 2},
                {"question": "What's the SLA on our payment provider?",
                 "answer": "I don't know", "category": 5, "evidence_session": None},
            ],
        },
        {
            "id": "demo-2-infra",
            "speaker_a": "Riley",
            "speaker_b": "Jess",
            "sessions": [
                {
                    "session_id": 1,
                    "date": "2026-03-30",
                    "messages": [
                        {"speaker": "Riley", "content": "auth-svc is written in Go 1.22 and runs on port 8000."},
                        {"speaker": "Jess", "content": "Noted: auth-svc uses Go 1.22, port 8000."},
                        {"speaker": "Riley", "content": "We use Redis 7.2 for session cache, hosted on Upstash."},
                        {"speaker": "Jess", "content": "Got it — Redis 7.2 on Upstash for sessions."},
                    ],
                },
                {
                    "session_id": 2,
                    "date": "2026-04-18",
                    "messages": [
                        {"speaker": "Riley", "content": "We upgraded auth-svc to Go 1.23 last sprint."},
                        {"speaker": "Jess", "content": "Noted — auth-svc is now on Go 1.23."},
                        {"speaker": "Riley", "content": "Migrated session cache from Upstash to AWS ElastiCache last Friday."},
                        {"speaker": "Jess", "content": "Understood — sessions are now on AWS ElastiCache."},
                    ],
                },
                {
                    "session_id": 3,
                    "date": "2026-05-10",
                    "messages": [
                        {"speaker": "Riley", "content": "Just rolled out OpenTelemetry tracing. Backend is Honeycomb."},
                        {"speaker": "Jess", "content": "Got it — tracing via OpenTelemetry to Honeycomb."},
                    ],
                },
            ],
            "qa": [
                {"question": "What language is auth-svc written in?", "answer": "Go 1.23",
                 "category": 1, "evidence_session": 2},
                {"question": "Where do we host our session cache?", "answer": "AWS ElastiCache",
                 "category": 1, "evidence_session": 2},
                {"question": "What's our tracing stack?", "answer": "OpenTelemetry with Honeycomb",
                 "category": 1, "evidence_session": 3},
                {"question": "Did our auth language and session cache both change?",
                 "answer": "Yes, auth went from Go 1.22 to Go 1.23 and session cache moved from Upstash to AWS ElastiCache",
                 "category": 2, "evidence_session": 2},
                {"question": "What is auth-svc's CPU usage?", "answer": "I don't know",
                 "category": 5, "evidence_session": None},
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# LOCOMO native-format loader. The HuggingFace dataset format puts each
# session under a key like ``session_1``, ``session_2``, etc. Adapt it
# into our canonical internal shape (matching BUILTIN_SAMPLE).
# ---------------------------------------------------------------------------


def _normalize_locomo_native(raw: dict | list) -> dict:
    """Try to coerce a real LOCOMO JSON dump into our canonical shape."""
    if isinstance(raw, dict) and "conversations" in raw:
        return raw  # already canonical
    if isinstance(raw, list):
        conversations = raw
    else:
        conversations = [raw]

    out_conversations: list[dict] = []
    for conv in conversations:
        conv_id = conv.get("sample_id") or conv.get("id") or f"conv-{len(out_conversations)+1}"
        conv_block = conv.get("conversation", conv)
        sessions: list[dict] = []
        # native: keys session_1, session_2, …
        session_keys = sorted(
            (k for k in conv_block if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        for sk in session_keys:
            n = int(sk.split("_")[1])
            messages = []
            for m in conv_block[sk]:
                content = m.get("text") or m.get("content") or ""
                speaker = m.get("speaker") or "speaker"
                messages.append({"speaker": speaker, "content": content})
            sessions.append({
                "session_id": n,
                "date": conv_block.get(f"{sk}_date_time"),
                "messages": messages,
            })
        out_conversations.append({
            "id": conv_id,
            "speaker_a": conv_block.get("speaker_a"),
            "speaker_b": conv_block.get("speaker_b"),
            "sessions": sessions,
            "qa": conv.get("qa", []),
        })
    return {"conversations": out_conversations}


def load_locomo(path: str | None) -> dict:
    if not path:
        return BUILTIN_SAMPLE
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return _normalize_locomo_native(raw)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "but",
    "we", "us", "our", "i", "you", "he", "she", "it", "they", "this",
    "that", "these", "those", "do", "does", "did", "have", "has", "had",
}


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", s.lower()) if w not in _STOPWORDS and len(w) > 1]


def context_recall(context: str, gold_answer: str) -> float:
    """Fraction of (non-stopword) gold-answer tokens that appear in context."""
    gold_toks = _tokens(gold_answer)
    if not gold_toks:
        return 1.0
    ctx_toks_set = set(_tokens(context))
    hits = sum(1 for t in gold_toks if t in ctx_toks_set)
    return hits / len(gold_toks)


def token_f1(predicted_text: str, gold_answer: str) -> float:
    pred = set(_tokens(predicted_text))
    gold = set(_tokens(gold_answer))
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = pred & gold
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred)
    recall = len(overlap) / len(gold)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# In-process pipeline builder. Uses a TEMP store so the user's real
# ~/.gml/memories.jsonl is never touched.
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _build_pipeline_with_temp_store(
    use_entity_index: bool | None = None,
    use_hyde: bool | None = None,
):
    """Build a fresh pipeline + memory store for one bench conversation.

    Returns: (pipeline, retriever, store, tmp_path, aal_extractor, entity_index).
    Last two are None when their feature flags are off.

    Tier-2 toggles:
      - ``use_entity_index``: wrap HybridRetriever in EntityBoostedRetriever
        (default: env GML_ENTITY_INDEX or True)
      - ``use_hyde``: wrap embedder in HydeEmbedder
        (default: env GML_HYDE or True)
    """
    use_entity_index = bool(
        use_entity_index if use_entity_index is not None
        else (os.environ.get("GML_ENTITY_INDEX", "1") == "1")
    )
    use_hyde = bool(
        use_hyde if use_hyde is not None
        else HYDE_ENABLED_DEFAULT
    )

    config = load_config(_project_root() / "config" / "orchestration.toml")

    try:
        emb_model = os.environ.get("GML_EMBED_MODEL", "").strip()
        if emb_model and os.path.exists(emb_model):
            from orchestration.embedder import SentenceTransformerEmbedder
            base_embedder = SentenceTransformerEmbedder(
                model_name=emb_model,
                device=os.environ.get("GML_EMBED_DEVICE", "mps"),
            )
            print(f"• Embedder: SentenceTransformerEmbedder({emb_model})", file=sys.stderr)
        else:
            base_embedder = FastEmbedEmbedder()
            print(f"• Embedder: FastEmbedEmbedder({base_embedder.model_name})", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] Embedder init failed ({type(exc).__name__}); using StubEmbedder. "
              "Retrieval will be essentially random.", file=sys.stderr)
        base_embedder = StubEmbedder(dim=384)

    # Optional HyDE wrapper (B3) — applies hypothetical-answer rewriting
    # at the embed stage. Default to single-rewrite "fuse" mode; the
    # multi-paraphrase C9 variant DILUTES the embedding empirically (verified
    # on LOCOMO bench, -0.01 vs single), so off by default. Re-enable
    # with GML_HYDE_MULTI=1 if you want to A/B it.
    embedder = base_embedder
    if use_hyde:
        hyde_mode = "multi" if os.environ.get("GML_HYDE_MULTI", "0") == "1" else "fuse"
        try:
            embedder = HydeEmbedder(
                base_embedder, make_local_llm_client(),
                mode=hyde_mode, n_paraphrases=3,
            )
            print(f"• HyDE mode: {hyde_mode}", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] HyDE init failed ({type(exc).__name__}); using plain embedder", file=sys.stderr)

    tmp = Path(tempfile.mkstemp(prefix="locomo-bench-", suffix=".jsonl")[1])
    store = JsonlMemoryStore(tmp)

    # SemanticRetriever uses the BASE embedder, not the HyDE wrapper, because
    # we want documents indexed by their own embeddings — only the QUERY gets
    # HyDE rewrite. The Pipeline's `embedder` is used for queries.
    #
    # NOTE: bench currently uses in-memory retrievers regardless of
    # GML_STORAGE_BACKEND. To run against the Postgres backend, the call
    # site needs to become async + use make_hybrid_retriever(base_embedder)
    # — a small follow-up since _build_pipeline_with_temp_store is sync today.
    dense = SemanticRetriever(embedder=base_embedder)
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    # Optional Entity-Boost wrapper (B1) — maintains an entity hash-index
    # alongside the retriever and bumps similarity of entity-matching hits.
    entity_index = None
    if use_entity_index:
        entity_index = EntityIndex()
        retriever = EntityBoostedRetriever(hybrid, entity_index, boost=0.15)
    else:
        retriever = hybrid

    # Time-aware wrapper (B7+B8) — off by default. Empirically did NOT move
    # cat-2 (multi-hop temporal) recall on the LOCOMO bench: the date boost
    # is small relative to cross-encoder's [-10,+10] reranking power.
    # Date entities ARE still indexed in EntityIndex (B6) which works fine.
    # Re-enable explicitly with GML_TIME_AWARE=1 if you want to retest.
    use_time_aware = os.environ.get("GML_TIME_AWARE", "0") == "1"
    if use_time_aware:
        retriever = TimeAwareRetriever(retriever, date_boost=0.15)

    # Multi-hop-aware wrapper: sliding windows are useful for multi-hop
    # queries but dilute single-hop. This filters them out at retrieval
    # time unless classify_query.is_multi_hop is True. Default ON.
    if os.environ.get("GML_MULTIHOP_AWARE", "1") == "1":
        retriever = MultiHopAwareRetriever(retriever)

    sam = SAM.with_ollama()

    pipeline = Pipeline(
        classifier=KeywordClassifier(),
        embedder=embedder,
        retriever=retriever,
        reranker=make_reranker(config),
        sam=sam,
        translator=Translator(),
        config=config,
    )

    # AAL tuple extractor (B4) — created here for the caller to use during
    # ingest. We don't always run it (env GML_AAL_TUPLES gates it) but the
    # extractor itself is cheap to construct.
    aal_extractor = None
    if AAL_ENABLED_DEFAULT:
        try:
            aal_extractor = AALTupleExtractor(client=make_local_llm_client())
        except Exception as exc:
            print(f"[warn] AAL extractor init failed ({type(exc).__name__}); skipping AAL tuples", file=sys.stderr)

    return pipeline, retriever, store, tmp, aal_extractor, entity_index


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


async def ingest_session_sdp(
    session: dict, sdp: SDPPipeline, retriever: HybridRetriever,
    store: JsonlMemoryStore,
) -> int:
    """Run every (speaker_a, speaker_b) turn pair through SDP. Returns # memories added."""
    n_added = 0
    msgs = session["messages"]
    # Take adjacent pairs as (user, assistant). For LOCOMO's two-speaker
    # conversations this is the natural framing.
    for i in range(0, len(msgs) - 1, 2):
        user_msg = msgs[i].get("content", "")
        assistant_msg = msgs[i + 1].get("content", "") if i + 1 < len(msgs) else ""
        aal_mems = sdp.process_turn(user_msg, assistant_msg)
        if not aal_mems:
            continue
        items = [m.to_memory_item() for m in aal_mems]
        store.add_many(items)
        await retriever.ingest(items)
        n_added += len(items)
    return n_added


async def ingest_session_llm(
    session: dict, extractor: MemoryExtractor, retriever: HybridRetriever,
    store: JsonlMemoryStore,
) -> int:
    """Run every turn-pair through the LLM extractor. SLOW (~10s/turn-pair)."""
    n_added = 0
    msgs = session["messages"]
    for i in range(0, len(msgs) - 1, 2):
        user_msg = msgs[i].get("content", "")
        assistant_msg = msgs[i + 1].get("content", "") if i + 1 < len(msgs) else ""
        try:
            items = await extractor.extract(
                user_query=user_msg, assistant_reply=assistant_msg,
            )
        except Exception as exc:
            print(f"[warn] extractor failed on turn {i}: {type(exc).__name__}", file=sys.stderr)
            continue
        if not items:
            continue
        store.add_many(items)
        await retriever.ingest(items)
        n_added += len(items)
    return n_added


async def ingest_session_raw(
    session: dict, retriever: HybridRetriever, store: JsonlMemoryStore,
    window_size: int | None = None, dedup: bool | None = None,
) -> int:
    """Store every message as-is as a MemoryItem, plus sliding-window chunks.

    Two memory shapes per session:
      1. one MemoryItem per single message  (source="locomo-raw")
      2. one MemoryItem per N-message sliding window (source="locomo-window")
         — captures conversational context for multi-hop questions

    Optionally dedups near-duplicate pleasantries via MinHash LSH before
    indexing (B5).
    """
    import uuid
    from datetime import datetime, timezone
    from orchestration.pipeline.contracts import MemoryItem
    from orchestration.sdp.date_resolver import enrich_with_resolved_dates
    from orchestration.sdp.dedup import MinHashDeduper

    # Defaults — proven to help stays default-on; dilutive features default-off.
    use_date_enrich = os.environ.get("GML_DATE_ENRICH", "1") == "1"
    if window_size is None:
        # Sliding windows are now safe to enable: MultiHopAwareRetriever
        # gates them out of single-hop queries' candidate pool but lets
        # them through on multi-hop queries (where they actually help).
        # Default size 3 (3-message overlapping windows). GML_SLIDING_WINDOWS=1 disables.
        window_size = int(os.environ.get("GML_SLIDING_WINDOWS", "3"))
    if dedup is None:
        # MinHash dedup catches almost nothing on LOCOMO (no exact duplicates).
        # Off by default; turn on with GML_MINHASH_DEDUP=1.
        dedup = os.environ.get("GML_MINHASH_DEDUP", "0") == "1"

    items: list[MemoryItem] = []
    sess_id = session.get("session_id")
    sess_date = session.get("date")
    sess_ts = _resolve_session_ts(sess_date)

    msgs = session.get("messages", [])
    # Stage 1: individual messages
    msg_items: list[MemoryItem] = []
    for m in msgs:
        text = m.get("content") or m.get("text") or ""
        if not text:
            continue
        speaker = m.get("speaker") or "speaker"
        dia_id = m.get("dia_id")
        content = f"{speaker}: {text}"
        # Phase #6 date arithmetic: enrich content with resolved-date
        # annotations. "Sunday" + session_ts(Wed May 10) → appended
        # "[resolved: 2023-05-07 (Sunday May 7 2023)]".
        if use_date_enrich:
            content = enrich_with_resolved_dates(content, anchor=sess_ts)
        msg_items.append(MemoryItem(
            id=f"raw-{uuid.uuid4().hex[:12]}",
            content=content,
            summary_short=text[:120],
            entity=speaker,
            attribute=None,
            value=None,
            timestamp=sess_ts,
            source="locomo-raw",
            authority_score=0.7,
            pinned=False,
            raw_metadata={"session_id": sess_id, "dia_id": dia_id},
        ))

    # Optional MinHash dedup of msg_items — drops near-duplicate pleasantries
    if dedup and len(msg_items) > 1:
        deduper = MinHashDeduper(threshold=0.75)
        keep = set(deduper.filter_unique([m.content for m in msg_items]))
        msg_items = [m for i, m in enumerate(msg_items) if i in keep]

    items.extend(msg_items)

    # Stage 2: sliding 3-message windows (overlap by stride 1)
    if window_size > 1 and len(msgs) >= window_size:
        for i in range(len(msgs) - window_size + 1):
            chunk = msgs[i : i + window_size]
            texts = []
            for m in chunk:
                txt = m.get("content") or m.get("text") or ""
                spk = m.get("speaker") or "speaker"
                if txt:
                    texts.append(f"{spk}: {txt}")
            if not texts:
                continue
            content = " | ".join(texts)
            if use_date_enrich:
                content = enrich_with_resolved_dates(content, anchor=sess_ts)
            items.append(MemoryItem(
                id=f"win-{uuid.uuid4().hex[:12]}",
                content=content,
                summary_short=content[:160],
                entity=None,
                attribute=None,
                value=None,
                timestamp=sess_ts,
                source="locomo-window",
                authority_score=0.65,  # slightly lower than single-msg
                pinned=False,
                raw_metadata={
                    "session_id": sess_id,
                    "window_start": chunk[0].get("dia_id"),
                    "window_end": chunk[-1].get("dia_id"),
                    "window_size": window_size,
                },
            ))

    if not items:
        return 0
    store.add_many(items)
    await retriever.ingest(items)
    return len(items)


# NOTE: llama.cpp serializes inference internally. Sending 8 concurrent
# requests doesn't get 8x throughput — it gets a queue, and our client
# timeout fires before the back of the queue gets processed. Default to
# 2 (one buffered request while another runs). Set GML_AAL_CONCURRENCY=1
# for fully serial behavior on flaky setups.
_AAL_CONCURRENCY = int(os.environ.get("GML_AAL_CONCURRENCY", "2"))


async def ingest_session_summary(
    session: dict, compressor: SAMTurnCompressor, writer: SDPWriter,
) -> int:
    """Phase #2: emit one per-session topic summary memory.

    SAM reads the whole session, produces a single ~30-word summary +
    topic + entities + importance, and SDPWriter persists it with
    source="session-summary" (authority 0.80). Used to boost cat-4
    (open-domain) which needs topic-level retrieval.
    """
    from datetime import datetime, timezone
    msgs = session.get("messages", [])
    if not msgs:
        return 0
    sess_id = session.get("session_id")
    sess_date = session.get("date")
    sess_ts = _resolve_session_ts(sess_date)
    try:
        rec = await compressor.summarize_session(
            msgs, timestamp=sess_ts, session_id=sess_id,
        )
    except Exception as exc:
        print(f"[warn] session-summary failed: {type(exc).__name__}", file=sys.stderr)
        return 0
    if rec.is_empty:
        return 0
    items = await writer.write_session_summary(rec)
    return len(items)


async def ingest_session_sam_aal(
    session: dict, compressor: SAMTurnCompressor, writer: SDPWriter,
) -> int:
    """SAM→AAL→SDP per-turn ingest (the canonical architecture).

    For each (user, assistant) turn-pair: SAM compresses into an AALRecord,
    SDPWriter persists the tuples + chunk + entities. Async-parallelized.
    """
    from datetime import datetime, timezone

    msgs = session.get("messages", [])
    sess_id = session.get("session_id")
    sess_date = session.get("date")
    sess_ts = _resolve_session_ts(sess_date)

    sem = asyncio.Semaphore(_AAL_CONCURRENCY)

    async def _compress_pair(i: int, u: dict, a: dict) -> AALRecord | None:
        u_text = u.get("content") or u.get("text") or ""
        a_text = a.get("content") or a.get("text") or ""
        if not u_text and not a_text:
            return None
        async with sem:
            try:
                return await compressor.compress(
                    user_text=u_text, assistant_text=a_text,
                    timestamp=sess_ts, session_id=sess_id,
                    dia_id_user=u.get("dia_id"), dia_id_assistant=a.get("dia_id"),
                )
            except Exception as exc:
                print(f"[warn] SAM compress failed on turn {i}: {type(exc).__name__}",
                      file=sys.stderr)
                return None

    tasks = []
    for i in range(0, len(msgs) - 1, 2):
        u = msgs[i]
        a = msgs[i + 1] if i + 1 < len(msgs) else {}
        tasks.append(_compress_pair(i, u, a))

    records = await asyncio.gather(*tasks)
    n_items = 0
    for rec in records:
        if rec is None or rec.is_empty:
            continue
        try:
            items = await writer.write(rec)
            n_items += len(items)
        except Exception as exc:
            print(f"[warn] SDPWriter failed: {type(exc).__name__}", file=sys.stderr)
    return n_items


async def ingest_session_aal(
    session: dict, aal_extractor: AALTupleExtractor,
    retriever: HybridRetriever, store: JsonlMemoryStore,
) -> int:
    """B4 (with A2 async optimization): extract AAL tuples per turn-pair in parallel.

    Each turn-pair requires an LLM call (~1-3s on Qwen3.5-4B-Q4). With
    serial ingest, a 30-session conversation = 300 turn-pairs × 2s = 10 min.
    With ``asyncio.gather`` + Semaphore(8) we get 4-8x throughput.

    The LLM server (llama.cpp) handles concurrent requests fine; the cap
    keeps us from blowing past its queue or starving SAM at query time.
    """
    from datetime import datetime, timezone

    msgs = session.get("messages", [])
    sess_id = session.get("session_id")
    sess_date = session.get("date")
    sess_ts = _resolve_session_ts(sess_date)

    sem = asyncio.Semaphore(_AAL_CONCURRENCY)

    async def _extract_pair(i: int, u: dict, a: dict):
        u_text = u.get("content") or u.get("text") or ""
        a_text = a.get("content") or a.get("text") or ""
        if not u_text and not a_text:
            return []
        async with sem:
            try:
                return await aal_extractor.extract_from_turn(
                    user_text=u_text, assistant_text=a_text, timestamp=sess_ts,
                    session_id=sess_id,
                    dia_id_user=u.get("dia_id"), dia_id_assistant=a.get("dia_id"),
                )
            except Exception as exc:
                print(f"[warn] AAL extract failed on turn {i}: {type(exc).__name__}",
                      file=sys.stderr)
                return []

    tasks = []
    for i in range(0, len(msgs) - 1, 2):
        u = msgs[i]
        a = msgs[i + 1] if i + 1 < len(msgs) else {}
        tasks.append(_extract_pair(i, u, a))

    results = await asyncio.gather(*tasks)

    # Flatten and write in deterministic order so the index is the same
    # regardless of concurrency completion order.
    all_items = [item for group in results for item in group]
    if not all_items:
        return 0
    store.add_many(all_items)
    await retriever.ingest(all_items)
    return len(all_items)


async def run_conversation(
    conv: dict, ingest_mode: str, label_prefix: str = "",
    verbose: bool = True, checkpoint_path: Path | None = None,
) -> dict:
    """Run one LOCOMO-style conversation: ingest + answer QAs + score."""
    pipeline, retriever, store, tmp_path, aal_extractor, entity_index = (
        _build_pipeline_with_temp_store()
    )
    sdp = SDPPipeline(source_tag="locomo") if ingest_mode == "sdp" else None
    extractor = MemoryExtractor(client=make_local_llm_client()) if ingest_mode == "llm" else None

    # SAM→AAL→SDP wiring. Used by both "sam-aal" (per-turn + summary)
    # and "sam-summary" (per-session summary only, the LEAN canonical path).
    sam_compressor: SAMTurnCompressor | None = None
    sdp_writer: SDPWriter | None = None
    if ingest_mode in ("sam-aal", "sam-summary"):
        try:
            sam_compressor = SAMTurnCompressor(client=make_local_llm_client())
            sdp_writer = SDPWriter(
                store=store, retriever=retriever, entity_index=entity_index,
            )
        except Exception as exc:
            print(f"[warn] SAM-SDP init failed ({type(exc).__name__}); "
                  f"falling back to raw mode", file=sys.stderr)
            ingest_mode = "raw"

    # #5: Optional answer generator. Lets the bench measure F1 (vs gold
    # answer) in addition to context_recall. Gated by GML_ANSWER_GEN=1.
    # When GML_ANSWER_LLM_BACKEND is set, answer-gen uses a separate LLM
    # (e.g. gemma2:27b) from the ingest LLM (FT'd Qwen).
    answer_gen: AnswerGenerator | None = None
    if ANSWER_GEN_ENABLED_DEFAULT:
        try:
            from orchestration.sam._ollama_client import make_answer_llm_client
            ans_client, uses_ft = make_answer_llm_client()
            answer_gen = AnswerGenerator(client=ans_client, uses_ft_prompt=uses_ft)
            ans_backend = os.environ.get("GML_ANSWER_LLM_BACKEND", "").strip().lower() or "(shared with ingest)"
            ans_model = (
                os.environ.get("GML_ANSWER_OLLAMA_MODEL")
                or os.environ.get("GML_OLLAMA_MODEL", "")
                or "n/a"
            )
            print(f"• Answer generation: enabled — backend={ans_backend}, model={ans_model}, ft_prompt={uses_ft}", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] AnswerGenerator init failed ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    conv_id = conv.get("id", "?")
    sessions = conv.get("sessions", [])
    n_msgs = sum(len(s.get("messages", [])) for s in sessions)
    qas_full = conv.get("qa", [])
    # Optionally cap QAs to bound runtime
    cap = int(globals().get("_MAX_QA_PER_CONV", 0))
    qas = qas_full[:cap] if cap > 0 else qas_full
    conv = {**conv, "qa": qas}
    n_qa = len(qas)
    print(f"{label_prefix}── {conv_id} ──  {len(sessions)} sessions, {n_msgs} msgs, "
          f"{n_qa} QA ({len(qas_full)} total)", flush=True)

    # ---- Ingest -----------------------------------------------------
    t0 = time.perf_counter()
    n_mem = 0
    for sess in sessions:
        if ingest_mode == "llm":
            n_mem += await ingest_session_llm(sess, extractor, retriever, store)
        elif ingest_mode == "raw":
            n_mem += await ingest_session_raw(sess, retriever, store)
        elif ingest_mode == "raw+aal":
            n_mem += await ingest_session_raw(sess, retriever, store)
            if aal_extractor is not None:
                n_mem += await ingest_session_aal(sess, aal_extractor, retriever, store)
        elif ingest_mode == "sam-aal":
            # Canonical SAM→AAL→SDP path. Three memory shapes per session:
            #   - raw messages + sliding-window chunks  (B2 / B5 baseline)
            #   - SAM-compressed tuples + chunks per turn (B4 + sam-aal)
            #   - one per-session topic summary           (Phase #2)
            n_mem += await ingest_session_raw(sess, retriever, store)
            if sam_compressor is not None and sdp_writer is not None:
                n_mem += await ingest_session_sam_aal(sess, sam_compressor, sdp_writer)
                n_mem += await ingest_session_summary(sess, sam_compressor, sdp_writer)
        elif ingest_mode == "sam-summary":
            # LEAN canonical path. ONE LLM call per session (the summary)
            # instead of N per-turn calls. The audit showed per-turn AAL
            # extraction times out 75% of the time under llama.cpp
            # serialization — net effect was empty tuples anyway. This
            # mode skips per-turn AAL entirely and relies on:
            #   - raw messages with date enrichment   (cat-2 lever)
            #   - per-session topic summary           (cat-4 lever)
            n_mem += await ingest_session_raw(sess, retriever, store)
            if sam_compressor is not None and sdp_writer is not None:
                n_mem += await ingest_session_summary(sess, sam_compressor, sdp_writer)
        else:  # sdp
            n_mem += await ingest_session_sdp(sess, sdp, retriever, store)
    # ── Tier 3.2: entity synthesis (post-ingest aggregation) ──────────
    # After all sessions for the conv are in, scan memories and emit
    # one synth memory per top entity. Single retrieval can then surface
    # multi-fact answers about a single entity in top-K. Default ON;
    # disable with GML_ENTITY_SYNTH=0.
    suffix_synth = ""
    if (
        os.environ.get("GML_ENTITY_SYNTH", "1") == "1"
        and entity_index is not None
        and entity_index.entity_count > 0
    ):
        from orchestration.sdp.entity_synth import synthesize_entity_memories
        try:
            all_mems = store.load_all()
            top_ents = entity_index.top_entities(n=20)
            synths = synthesize_entity_memories(
                all_mems, top_entities=top_ents, max_entities=8,
            )
            if synths:
                store.add_many(synths)
                await retriever.ingest(synths)
                n_mem += len(synths)
                suffix_synth = f", {len(synths)} entity synths"
        except Exception as exc:
            print(f"  [warn] entity_synth failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    ingest_ms = int((time.perf_counter() - t0) * 1000)
    suffix = ""
    if entity_index is not None:
        suffix += f", {entity_index.entity_count} entities"
    suffix += suffix_synth
    print(f"  ingest [{ingest_mode}]: {n_mem} memories in {ingest_ms/1000:.1f}s{suffix}", flush=True)

    # ---- QA ---------------------------------------------------------
    results = []
    target = TargetDescriptor.for_claude()
    qas = conv.get("qa", [])
    cat_acc: dict[int, list[float]] = defaultdict(list)
    qa_t0 = time.perf_counter()

    for idx, qa in enumerate(qas, start=1):
        q_text = qa.get("question") or ""
        if not q_text:
            continue
        # LOCOMO category-5 (adversarial) QAs carry `adversarial_answer` instead of `answer`.
        gold = qa.get("answer") or qa.get("adversarial_answer") or ""
        if not gold:
            continue
        category = qa.get("category", 1)

        q = Pipeline.build_query(text=q_text, target=target)
        t0 = time.perf_counter()
        payload = await pipeline.run(q)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        ctx = payload.formatted_context

        # LOCOMO cat 5 (adversarial) carries `adversarial_answer` as the
        # actual correct response — it's NOT a refusal-target. Score it
        # the same as the other categories.
        recall = context_recall(ctx, str(gold))
        f1 = token_f1(ctx, str(gold))

        # Retrieval-stage recall@K vs gold evidence sessions. Real LOCOMO QAs
        # use ``evidence: ["D1:3", ...]`` (dia_id format); the synthetic
        # sample uses ``evidence_session: 2``. Support both.
        gold_sessions: set[int] = set()
        ev = qa.get("evidence")
        if isinstance(ev, list):
            for d in ev:
                if isinstance(d, str) and d.startswith("D") and ":" in d:
                    try:
                        gold_sessions.add(int(d.split(":", 1)[0][1:]))
                    except ValueError:
                        pass
        es = qa.get("evidence_session")
        if es is not None:
            try:
                gold_sessions.add(int(es))
            except (TypeError, ValueError):
                pass

        top_sess_ids = payload.metadata.get("top_session_ids") if payload.metadata else None
        hit_at_5 = None
        if gold_sessions and top_sess_ids:
            try:
                top5_set = {int(s) for s in top_sess_ids[:5]}
                hit_at_5 = 1 if gold_sessions & top5_set else 0
            except (TypeError, ValueError):
                hit_at_5 = None

        # #5: optional answer generation + F1 vs gold. This is the metric
        # LOCOMO papers report directly.
        generated_answer = None
        answer_f1 = None
        if answer_gen is not None:
            try:
                generated_answer = await answer_gen.answer(ctx, q_text, category=category)
                answer_f1 = token_f1(generated_answer, str(gold))
            except Exception as exc:
                print(f"[warn] answer gen failed: {type(exc).__name__}", file=sys.stderr)

        cat_acc[category].append(recall)
        record = {
            "question": q_text, "gold": gold, "category": category,
            "recall": recall, "f1": f1, "ms": dur_ms,
        }
        if generated_answer is not None:
            record["generated_answer"] = generated_answer
            record["answer_f1"] = answer_f1
        if hit_at_5 is not None:
            record["hit_at_5"] = hit_at_5
            record["gold_sessions"] = sorted(gold_sessions)
            record["top_session_ids"] = top_sess_ids[:5] if top_sess_ids else []
        results.append(record)

        if verbose and idx % 20 == 0:
            elapsed = time.perf_counter() - qa_t0
            rate = idx / max(elapsed, 1e-6)
            eta = (len(qas) - idx) / max(rate, 1e-6)
            running_recall = sum(r["recall"] for r in results) / len(results)
            print(f"    {idx}/{len(qas)} QA · running recall={running_recall:.2f} · "
                  f"rate={rate:.1f}/s · ETA={eta/60:.1f}m", flush=True)

    qa_ms = int((time.perf_counter() - qa_t0) * 1000)

    # Per-conversation summary line
    per_cat_summary = ", ".join(
        f"c{cat}:{sum(rs)/len(rs):.2f}({len(rs)})"
        for cat, rs in sorted(cat_acc.items())
    )
    overall = sum(r["recall"] for r in results) / max(len(results), 1)
    print(f"  done in {qa_ms/1000:.0f}s · recall={overall:.2f} · {per_cat_summary}", flush=True)

    # Cleanup temp file
    try:
        tmp_path.unlink()
    except OSError:
        pass

    run = {
        "conv_id": conv_id, "n_mem": n_mem, "ingest_ms": ingest_ms,
        "qa_ms": qa_ms, "qa_results": results,
    }

    # Persist checkpoint (incremental)
    if checkpoint_path is not None:
        try:
            existing = json.loads(checkpoint_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"runs": []}
        existing["runs"].append(run)
        checkpoint_path.write_text(json.dumps(existing, indent=1, default=str))

    return run


def summarize(all_runs: list[dict], ingest_mode_label: str) -> None:
    print(f"\n{'=' * 84}")
    print(f"LOCOMO LEADERBOARD  ingest-mode={ingest_mode_label}")
    print("=" * 84)

    # Aggregate per-category
    cat_recall: dict[int, list[float]] = defaultdict(list)
    cat_f1: dict[int, list[float]] = defaultdict(list)
    cat_label = {
        1: "single-hop", 2: "multi-hop", 3: "temporal",
        4: "open-domain", 5: "adversarial-refusal",
    }
    total_qa = 0
    total_ingest_ms = 0
    total_mem = 0

    for run in all_runs:
        total_ingest_ms += run["ingest_ms"]
        total_mem += run["n_mem"]
        for r in run["qa_results"]:
            total_qa += 1
            cat_recall[r["category"]].append(r["recall"])
            cat_f1[r["category"]].append(r["f1"])

    print(f"  conversations:   {len(all_runs)}")
    print(f"  memories total:  {total_mem}")
    print(f"  ingest time:     {total_ingest_ms/1000:.1f}s "
          f"({(total_ingest_ms/max(total_mem,1)):.0f}ms/memory)")
    print(f"  QA count:        {total_qa}")
    print()
    # Build per-category answer-F1 distribution too, when present
    cat_ans_f1: dict[int, list[float]] = defaultdict(list)
    for run in all_runs:
        for r in run["qa_results"]:
            if r.get("answer_f1") is not None:
                cat_ans_f1[r["category"]].append(r["answer_f1"])
    has_answer_f1 = any(cat_ans_f1.values())

    if has_answer_f1:
        print(f"  {'category':<22} {'n':>4} {'recall':>8} {'ctx_f1':>8} {'ans_f1':>8}")
    else:
        print(f"  {'category':<22} {'n':>4} {'recall':>8} {'f1':>8}")
    print("  " + "-" * (60 if has_answer_f1 else 50))
    for cat in sorted(set(list(cat_recall.keys()) + list(cat_f1.keys()))):
        rs = cat_recall.get(cat, [])
        fs = cat_f1.get(cat, [])
        if not rs:
            continue
        r_avg = sum(rs) / len(rs)
        f_avg = sum(fs) / len(fs)
        if has_answer_f1:
            ans = cat_ans_f1.get(cat, [])
            ans_avg = sum(ans) / len(ans) if ans else 0.0
            print(f"  {cat_label.get(cat, '?'):<22} {len(rs):>4} {r_avg:>8.2f} {f_avg:>8.2f} {ans_avg:>8.2f}")
        else:
            print(f"  {cat_label.get(cat, '?'):<22} {len(rs):>4} {r_avg:>8.2f} {f_avg:>8.2f}")
    print()
    all_recall = [v for vs in cat_recall.values() for v in vs]
    all_f1 = [v for vs in cat_f1.values() for v in vs]
    if all_recall:
        line = (
            f"  OVERALL              {len(all_recall):>4} "
            f"{sum(all_recall)/len(all_recall):>8.2f} "
            f"{sum(all_f1)/len(all_f1):>8.2f}"
        )
        if has_answer_f1:
            all_ans = [v for vs in cat_ans_f1.values() for v in vs]
            line += f" {sum(all_ans)/max(len(all_ans),1):>8.2f}"
        print(line)

    # Retrieval-stage recall@5 (when GML_BENCH_TRACE_HITS=1 captured hits)
    hits_at_5: list[int] = []
    cat_hits: dict[int, list[int]] = defaultdict(list)
    for run in all_runs:
        for r in run["qa_results"]:
            h = r.get("hit_at_5")
            if h is not None:
                hits_at_5.append(h)
                cat_hits[r["category"]].append(h)
    if hits_at_5:
        print()
        print(f"  retrieval recall@5 (evidence_session in top-5):")
        for cat in sorted(cat_hits.keys()):
            hs = cat_hits[cat]
            print(f"    {cat_label.get(cat, '?'):<22} {len(hs):>4} {sum(hs)/len(hs):>8.2f}")
        print(f"    {'OVERALL':<22} {len(hits_at_5):>4} {sum(hits_at_5)/len(hits_at_5):>8.2f}")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LOCOMO benchmark on the GML pipeline.")
    p.add_argument("--data", help="Path to LOCOMO JSON. If omitted, uses the built-in sample.")
    p.add_argument("--limit", type=int, default=0,
                   help="Run only the first N conversations (default: all).")
    p.add_argument("--ingest-mode",
                   choices=["sdp", "llm", "raw", "raw+aal", "sam-aal", "sam-summary", "both"],
                   default="sam-summary",
                   help="How to ingest each session's turns. sam-summary=raw + "
                        "per-session topic summary (LEAN canonical, default); "
                        "sam-aal=sam-summary + per-turn LLM extraction (slow); "
                        "raw=store every message only; "
                        "raw+aal=raw + standalone AAL tuples (legacy); "
                        "sdp=fast regex extraction; llm=full LLM extraction.")
    p.add_argument("--max-qa-per-conv", type=int, default=0,
                   help="Cap QAs per conversation. 0 = no cap (full LOCOMO).")
    p.add_argument("--checkpoint", help="Path to write per-conversation results to as we go.")
    return p.parse_args()


async def run_one_mode(data: dict, ingest_mode: str,
                       checkpoint_path: Path | None = None) -> list[dict]:
    runs = []
    convs = data.get("conversations", [])
    # Reset checkpoint at start of a mode
    if checkpoint_path is not None:
        checkpoint_path.write_text(json.dumps({"runs": []}, indent=1))
    for i, conv in enumerate(convs, start=1):
        run = await run_conversation(
            conv, ingest_mode=ingest_mode,
            label_prefix=f"[{ingest_mode} {i}/{len(convs)}] ",
            checkpoint_path=checkpoint_path,
        )
        runs.append(run)
        # Print running aggregate after each conv so progress is visible
        if i < len(convs):
            n_qa = sum(len(r["qa_results"]) for r in runs)
            avg_recall = (
                sum(qr["recall"] for r in runs for qr in r["qa_results"])
                / max(n_qa, 1)
            )
            print(f"  ── running totals: {i}/{len(convs)} convs · {n_qa} QA · "
                  f"recall={avg_recall:.3f}", flush=True)
    return runs


async def main() -> int:
    # Route structured pipeline logs to stderr so they don't drown progress output
    _set_log_stream(sys.stderr)
    # Quiet stdlib logging too (Ollama httpx INFO lines etc.)
    import logging
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "httpcore", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = parse_args()
    globals()["_MAX_QA_PER_CONV"] = args.max_qa_per_conv
    data = load_locomo(args.data)
    if args.limit:
        data = {"conversations": data["conversations"][: args.limit]}

    print(f"LOCOMO benchmark — {len(data['conversations'])} conversation(s)")
    print(f"  data source: {args.data or 'BUILT-IN SAMPLE'}")
    print(f"  ingest mode: {args.ingest_mode}")
    print(f"  pipeline:    classify→embed→retrieve→[SAM | rerank+SAM]→assemble→translate")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if args.ingest_mode == "both":
        sdp_cp = checkpoint_path.with_suffix(".sdp.json") if checkpoint_path else None
        llm_cp = checkpoint_path.with_suffix(".llm.json") if checkpoint_path else None
        sdp_runs = await run_one_mode(data, "sdp", checkpoint_path=sdp_cp)
        summarize(sdp_runs, "sdp")
        llm_runs = await run_one_mode(data, "llm", checkpoint_path=llm_cp)
        summarize(llm_runs, "llm")
    else:
        runs = await run_one_mode(data, args.ingest_mode, checkpoint_path=checkpoint_path)
        summarize(runs, args.ingest_mode)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
