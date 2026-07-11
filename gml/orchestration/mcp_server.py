"""GML as an MCP server — pluggable into Claude Desktop, Cursor, VS Code,
or any other MCP client.

This is the production usecase: instead of the user calling GML's CLI/HTTP,
the AI itself calls GML's tools when it needs memory. The user types into
ChatGPT/Claude/Cursor; the AI decides "I should check memory" and invokes
one of our tools; the AI uses the returned context to answer.

Two primary tools exercise the full designed pipeline:

  - query(text)
      Run the WHOLE pipeline: Classifier → Embedder → Retriever probe →
      branch (NO=SAM.reason_from_scratch / YES=top50→Reranker→SAM
      .resolve_conflicts) → Assembler → Translator. Returns formatted
      context the AI should consume to answer. THIS is the tool Claude
      Desktop should call BEFORE answering every user turn.

  - ingest(user_query, assistant_reply)
      Run MemoryExtractor on the exchange (local DeepSeek R1 8B by
      default), persist extracted MemoryItems to ~/.gml/memories.jsonl,
      and live-ingest them into the retriever so the NEXT query call
      sees them. Claude Desktop should call this AFTER answering.

Six lower-level tools remain for direct/debug access:

  - recall, remember, forget, list_memories, improve_query, status

Run over stdio (default for Claude Desktop / Cursor):

    gml mcp

Or programmatically:

    python -m orchestration.mcp_server
"""
import asyncio
import math
import contextvars
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Per-request user context.
#
# The auth middleware (streamable-http transport) writes the resolved user_id
# here at the start of each request. MCP tool functions read it via
# ``_current_user_id()`` when they need to scope a query or memory write to
# the calling user. None means "admin / unscoped" (stdio transport, master
# key, or auth-disabled dev mode).
#
# ContextVar isolates concurrent requests automatically (each event-loop
# task gets its own value).
# ---------------------------------------------------------------------------
_CURRENT_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "gml_mcp_current_user_id", default=None,
)


def _current_user_id() -> str | None:
    uid = _CURRENT_USER_ID.get()
    if uid is None:
        # stdio transport (e.g. behind the relay connector) has no per-request
        # user. Fall back to a configured default tenant so memory WRITES work
        # (the Postgres backend requires a user_id). Reads scope to it too.
        uid = os.environ.get("GML_MCP_USER") or None
    return uid

from orchestration.classifier import KeywordClassifier
from orchestration.embedder import FastEmbedEmbedder, GeminiEmbedder, OllamaEmbedder, StubEmbedder
from orchestration.embedder.base import Embedder
from orchestration.ingestion import MemoryExtractor
from orchestration.memory_store import JsonlMemoryStore
from orchestration.sdp import SDPPipeline
from orchestration.observability.logging import set_output_stream as _set_log_stream
from orchestration.pipeline import (
    MemoryItem,
    Pipeline,
    TargetDescriptor,
    load_config,
)
from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    ResolvedMemorySet,
)
from orchestration.pipeline.pipeline import should_skip_sam
from orchestration.reranker import ScoreReranker, make_reranker
from orchestration.retriever import (
    BM25Retriever,
    HybridRetriever,
    SemanticRetriever,
    default_records,
)
from orchestration.sam import SAM
from orchestration.sam._ollama_client import (
    DEFAULT_BACKEND,
    health_probe_url,
    make_local_llm_client,
)
from orchestration.translator import Translator


# ---------------------------------------------------------------------------
# Embedder autodetect — same precedence the CLI uses.
# ---------------------------------------------------------------------------


def _autodetect_embedder() -> Embedder:
    """Pick the best available real Embedder.

    Order: FastEmbed (local ONNX, no daemon, no key) → Ollama
    (nomic-embed-text if pulled) → Gemini (if GEMINI_API_KEY set) →
    StubEmbedder with a stderr warning. The first three are real
    semantic embedders; the stub is hash-based and makes retrieval
    essentially random — we still return it so the server never refuses
    to start, but we shout about it.

    GML_EMBEDDER=st loads a local sentence-transformers model (the FT'd
    embedder at GML_ST_EMBED_MODEL). This MUST match what the HTTP API uses so
    query and stored vectors share one space in pgvector.
    """
    if os.environ.get("GML_EMBEDDER", "").strip().lower() == "st":
        from orchestration.embedder import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder(device=os.environ.get("GML_ST_DEVICE", "cpu"))
        sys.stderr.write(f"• embedder: SentenceTransformer {emb.model_name}\n")
        return emb
    try:
        emb = FastEmbedEmbedder()
        sys.stderr.write("• embedder: FastEmbed BAAI/bge-small-en-v1.5\n")
        return emb
    except Exception as exc:
        sys.stderr.write(f"• FastEmbed unavailable ({type(exc).__name__})\n")

    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        r.raise_for_status()
        names = {m.get("name", "").split(":")[0] for m in r.json().get("models", [])}
        if "nomic-embed-text" in names:
            sys.stderr.write("• embedder: Ollama nomic-embed-text\n")
            return OllamaEmbedder()
    except Exception:
        pass

    if os.environ.get("GEMINI_API_KEY"):
        sys.stderr.write("• embedder: Gemini gemini-embedding-001\n")
        return GeminiEmbedder()

    sys.stderr.write(
        "⚠ embedder: StubEmbedder (hash-based, retrieval will be random). "
        "Run `pip install fastembed` or `ollama pull nomic-embed-text`.\n"
    )
    return StubEmbedder(dim=384)


# ---------------------------------------------------------------------------
# Boot the full Pipeline + extractor (shared by every tool call).
# Lazy-initialized on first use to keep MCP startup fast — Claude Desktop
# times out servers that take too long to register.
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "config" / "orchestration.toml"


def _default_memory_path() -> Path:
    return Path(
        os.environ.get("GML_MEMORY_PATH")
        or (Path.home() / ".gml" / "memories.jsonl")
    )


class _State:
    def __init__(self) -> None:
        self.embedder: Embedder | None = None
        self.retriever: HybridRetriever | None = None
        self.store: JsonlMemoryStore | None = None
        self.sam: SAM | None = None
        self.pipeline: Pipeline | None = None
        self.extractor: MemoryExtractor | None = None
        self.sdp: SDPPipeline | None = None
        self.sam_llm_enabled: bool = False
        self.extractor_enabled: bool = False
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return

            config = load_config(_default_config_path())

            # Real semantic embedder — falls back loudly to stub.
            self.embedder = _autodetect_embedder()

            # Storage + retriever backends are picked by GML_STORAGE_BACKEND
            # via the storage factory. Same primitives the FastAPI server
            # uses (see orchestration.storage), so MCP and HTTP both write
            # to the SAME store under Postgres mode — no schism between
            # MCP-issued memories and HTTP-issued memories.
            from orchestration.storage import (
                _is_postgres, make_hybrid_retriever, make_memory_store,
            )
            self.store = await make_memory_store(
                fallback_jsonl_path=_default_memory_path(),
                embedder=self.embedder,
            )
            self.retriever = await make_hybrid_retriever(self.embedder)

            # JSONL backend keeps an in-memory retriever cache — seed it
            # from the store (or default fixture). Postgres backend reads
            # from the DB directly, no seeding needed.
            if not _is_postgres():
                seed = await self.store.load_all() or default_records()
                await self.retriever.ingest(seed)

            # SAM: try to wire the local-LLM reasoner. Backend chosen via
            # GML_LLM_BACKEND env var (default: llamacpp on port 8080).
            # If the LLM server is down, fall back to heuristic-only —
            # pipeline still completes.
            from orchestration.sam._ollama_client import DEFAULT_MODEL as _SAM_MODEL
            self.sam_backend = DEFAULT_BACKEND
            self.sam_model_name = _SAM_MODEL
            try:
                self.sam = SAM.with_ollama()
                # Probe so we detect a dead server during init, not mid-query.
                import httpx
                httpx.get(health_probe_url(), timeout=1.5).raise_for_status()
                self.sam_llm_enabled = True
                sys.stderr.write(
                    f"• SAM: {self.sam_backend} backend, model {self.sam_model_name} "
                    f"(LLM reasoning enabled)\n"
                )
            except Exception as exc:
                self.sam = SAM(reasoner=None)
                self.sam_llm_enabled = False
                sys.stderr.write(
                    f"⚠ SAM: {self.sam_backend} unreachable "
                    f"({type(exc).__name__}); heuristic-only reasoner. "
                    "`improve_query` and `reason_from_scratch` will be passthrough.\n"
                )

            # Memory extractor — same backend. If the LLM is down it will
            # silently return [] per its design; we degrade rather than break.
            try:
                self.extractor = MemoryExtractor(client=make_local_llm_client())
                self.extractor_enabled = self.sam_llm_enabled  # same dependency
            except Exception as exc:
                self.extractor = None
                self.extractor_enabled = False
                sys.stderr.write(
                    f"⚠ MemoryExtractor: init failed ({type(exc).__name__}); "
                    "ingest() will reject unless given explicit content.\n"
                )

            # Full Pipeline — same wiring as cli.py:_build_pipeline.
            self.pipeline = Pipeline(
                classifier=KeywordClassifier(),
                embedder=self.embedder,
                retriever=self.retriever,
                reranker=make_reranker(config),
                sam=self.sam,
                translator=Translator(),
                config=config,
            )

            # SDP — lightweight regex/heuristic ingestion. No LLM, no
            # extra deps. Used by sdp_ingest() as the fast path.
            self.sdp = SDPPipeline(source_tag="sdp")
            sys.stderr.write("• SDP: regex/heuristic pipeline ready (no LLM)\n")

            self._initialized = True


_state = _State()


# ---------------------------------------------------------------------------
# MCP server + tool definitions
# ---------------------------------------------------------------------------


mcp = FastMCP(
    name="gml-memory",
    instructions=(
        "GML is a long-term memory layer for AI assistants. Prefer the "
        "PIPELINE tools `query` and `ingest` over the low-level ones.\n\n"
        "BEFORE answering ANY user turn: call `query(text=<user's exact "
        "text>)`. It runs classify → embed → retrieve → SAM → assemble "
        "→ translate and returns the formatted context the user's "
        "memories add. Use that context to answer.\n\n"
        "AFTER answering: call `ingest(user_query=<original text>, "
        "assistant_reply=<your reply>)` so durable facts from the "
        "exchange get persisted and live-ingested for the next turn.\n\n"
        "Low-level tools (`recall`, `remember`, `forget`, `list_memories`, "
        "`improve_query`, `status`) are for direct/debug access only. "
        "Skip them in normal turns — `query`/`ingest` do the right thing."
    ),
    # HTTP transport bind config. Stdio ignores these.
    host=os.environ.get("GML_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("GML_MCP_PORT", "8765")),
    streamable_http_path=os.environ.get("GML_MCP_PATH", "/mcp"),
)


def _format_memories(records: list[dict[str, Any]]) -> str:
    """Human-readable summary the AI can paraphrase to the user."""
    if not records:
        return "(no relevant memories found)"
    lines = []
    for i, r in enumerate(records, start=1):
        head = f"{i}. [{r['source']}]"
        if r.get("entity"):
            head += f" {r['entity']}"
            if r.get("attribute"):
                head += f"/{r['attribute']}"
            if r.get("value"):
                head += f" = {r['value']}"
        lines.append(head)
        lines.append(f"   {r['content']}")
        if "similarity" in r:
            lines.append(f"   (relevance: {r['similarity']:.2f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primary tools — the ones Claude Desktop is meant to call.
# ---------------------------------------------------------------------------


@mcp.tool()
async def query(text: str = "", query: str = "") -> str:
    """Run the FULL GML pipeline for one user turn.

    Stages, in order: Classifier (intent) → Embedder (vector) →
    Retriever.search (probe) → branch on "found anything?":
      - NO  → SAM.reason_from_scratch (LLM with no memory context)
      - YES → Retriever.get_top_matches(50) → Reranker.pick_best(10) →
              SAM.resolve_conflicts (drops superseded memories)
    → Assembler.package (final 5) → Translator (formats for the target AI).

    Returns the formatted-context block the calling AI should paraphrase
    into its answer to the user. Also reports which branch was taken and
    how many memories made it into the context.

    Args:
        text: The user's question, verbatim. (Alias: ``query``.)
        query: Alias for ``text`` so callers consistent with the ``recall`` tool work too.
    """
    text = text or query
    if not text:
        return "error: provide the question as 'text' (or 'query')."
    await _state.init()
    target = TargetDescriptor.for_claude()
    q = Pipeline.build_query(text=text, target=target, user_id=_current_user_id())
    payload = await _state.pipeline.run(q)

    md = payload.metadata
    if md.get("short_circuit") == "pleasantry":
        branch = "pleasantry"
    elif md.get("reason_from_scratch"):
        branch = "no_match→SAM_from_scratch"
    else:
        branch = "match→rerank→SAM_resolve_conflicts"
    items = md.get("items_included")
    improved = md.get("query_was_improved")
    degraded_note = "" if _state.sam_llm_enabled else " sam=heuristic_only"

    return (
        f"<gml_context>\n{payload.formatted_context}\n</gml_context>\n"
        f"[branch={branch} items_included={items} "
        f"query_was_improved={improved}{degraded_note}]"
    )


@mcp.tool()
async def ingest(user_query: str, assistant_reply: str) -> str:
    """Persist durable facts from a completed user→assistant exchange.

    Runs MemoryExtractor (local DeepSeek R1 8B via Ollama) on the turn to
    pull out entity/attribute/value claims worth recalling later. Each
    extracted MemoryItem is persisted to ~/.gml/memories.jsonl AND
    live-ingested into the retriever so the next `query` call can see it.

    Call this AFTER your reply, with both sides of the exchange. If the
    extractor finds nothing (pure pleasantry, no facts), nothing is saved
    — that's by design, not an error.

    Args:
        user_query: What the user asked, verbatim.
        assistant_reply: What you just answered.
    """
    await _state.init()

    if _state.extractor is None:
        return "ingest unavailable: MemoryExtractor not initialized"

    extracted: list[MemoryItem] = []
    try:
        extracted = await _state.extractor.extract(
            user_query=user_query,
            assistant_reply=assistant_reply,
        )
    except Exception as exc:
        return f"ingest failed: {type(exc).__name__}: {exc}"

    if not extracted:
        deg = "" if _state.extractor_enabled else " (ollama unavailable — extractor returned empty)"
        return f"ingest: no durable facts extracted{deg}"

    await _state.store.add_many(extracted, user_id=_current_user_id())
    try:
        await _state.retriever.ingest(extracted)
    except Exception as exc:
        return (
            f"ingest: persisted {len(extracted)} memories but live-ingest "
            f"failed ({type(exc).__name__}: {exc}) — next query may not see them"
        )

    summary = ", ".join(f"{m.id}: {m.content[:60]!r}" for m in extracted[:5])
    more = "" if len(extracted) <= 5 else f" (+{len(extracted)-5} more)"
    return f"ingest: saved {len(extracted)} memories — {summary}{more}"


@mcp.tool()
async def sdp_ingest(user_query: str, assistant_reply: str) -> str:
    """Persist facts using the LIGHTWEIGHT regex-based SDP pipeline.

    Same idea as `ingest()` but uses the Semantic Decomposition Pipeline
    (no LLM): ConversationParser → SemanticExtractor → EntityExtractor →
    RelationshipMapper → ImportanceScorer + ConfidenceScorer →
    SemanticSummarizer → AALMemory → MemoryItem. Each AALMemory persists
    a SINGLE atomic semantic unit (the doc's "no paragraphs" rule).

    Trade-off vs `ingest()`:
      - sdp_ingest is ~100x FASTER (no LLM call — typically <50ms)
      - sdp_ingest catches ONLY pattern-detectable facts (tech stack,
        versions, ports, URLs, people, supersession verbs); paraphrased
        or implicit facts are missed
      - `ingest()` (LLM) catches more nuanced facts but pays ~7s per call

    Use `sdp_ingest` for the fast path on clearly-factual turns. Use
    `ingest()` when nuance matters and the latency is acceptable.

    Args:
        user_query: What the user said, verbatim.
        assistant_reply: What you just answered.
    """
    await _state.init()
    if _state.sdp is None:
        return "sdp_ingest unavailable: SDPPipeline not initialized"

    aal_memories = _state.sdp.process_turn(user_query, assistant_reply)
    if not aal_memories:
        return "sdp_ingest: no pattern-detectable facts in this turn"

    items = [m.to_memory_item() for m in aal_memories]
    await _state.store.add_many(items, user_id=_current_user_id())
    try:
        await _state.retriever.ingest(items)
    except Exception as exc:
        return (
            f"sdp_ingest: persisted {len(items)} memories but live-ingest "
            f"failed ({type(exc).__name__}: {exc})"
        )

    summary = ", ".join(f"{m.id}: {m.content[:60]!r}" for m in items[:5])
    more = "" if len(items) <= 5 else f" (+{len(items)-5} more)"

    # Pull a few headline scores so the AI calling this can sanity-check
    n_high_imp = sum(1 for m in aal_memories if m.importance >= 0.75)
    n_high_conf = sum(1 for m in aal_memories if m.confidence >= 0.9)
    n_entities = len({e["text"].lower() for m in aal_memories for e in m.entities})
    n_rels = len({(r["source"], r["relation"], r["target"]) for m in aal_memories for r in m.relationships})

    return (
        f"sdp_ingest: saved {len(items)} memories — {summary}{more}\n"
        f"  high-importance: {n_high_imp}/{len(items)}, "
        f"high-confidence: {n_high_conf}/{len(items)}, "
        f"entities: {n_entities}, relationships: {n_rels}"
    )


# ---------------------------------------------------------------------------
# Low-level tools — kept for direct/debug access. Skip in normal turns.
# ---------------------------------------------------------------------------


@mcp.tool()
async def recall(query: str, top_k: int = 5) -> str:
    """Low-level retrieval bypass. Skip the pipeline; just search the index.

    Prefer `query(text)` for normal use — it runs classification, SAM
    reasoning, reranking, and target-aware formatting. Use `recall` only
    when you explicitly need raw vector hits without those stages.

    Args:
        query: The user's question.
        top_k: How many memories to return. Default 5.
    """
    await _state.init()
    classification = Classification(
        intent_type="question",
        entities=[],
        retrieval_hints={},
        confidence=0.5,
        source=ClassificationSource.KEYWORD_FALLBACK,
    )
    target = TargetDescriptor.for_claude()
    q = Pipeline.build_query(query, target, user_id=_current_user_id())
    embedded = await _state.embedder.embed(q, classification)
    hits = await _state.retriever.get_top_matches(embedded, k=top_k)
    serialized = [
        {
            "id": h.record.id,
            "content": h.record.content,
            "source": h.record.source,
            "entity": h.record.entity,
            "attribute": h.record.attribute,
            "value": h.record.value,
            "similarity": h.similarity,
            "timestamp": h.record.timestamp.isoformat(),
        }
        for h in hits
    ]
    return _format_memories(serialized)


@mcp.tool()
async def remember(
    content: str,
    entity: str | None = None,
    attribute: str | None = None,
    value: str | None = None,
    source: str = "conversation",
    authority_score: float = 0.7,
) -> str:
    """Low-level save bypass. Persist one explicit fact, no extraction.

    Prefer `ingest(user_query, assistant_reply)` — it runs the LLM
    extractor which usually finds multiple structured facts per turn.
    Use `remember` only when you have a single, pre-formed claim and
    want to avoid the extractor pass.

    Args:
        content: One full-sentence claim worth remembering, third person.
        entity: Subject of the claim.
        attribute: Property of the entity.
        value: Value of the attribute.
        source: Where this came from. Default "conversation".
        authority_score: 0-1 trust score. Default 0.7.
    """
    await _state.init()
    item = MemoryItem(
        id=f"mcp-{uuid.uuid4().hex[:12]}",
        content=content,
        entity=entity,
        attribute=attribute,
        value=value,
        timestamp=datetime.now(timezone.utc),
        source=source,
        authority_score=max(0.0, min(1.0, authority_score)),
        pinned=False,
    )
    await _state.store.add(item, user_id=_current_user_id())
    await _state.retriever.ingest([item])
    return f"Saved memory {item.id}: {content!r}"


@mcp.tool()
async def forget(memory_id: str) -> str:
    """Remove a memory from the store.

    On the JSONL backend rewrites the file with the record dropped and
    rebuilds the in-memory retriever (BM25 has no incremental delete).
    On the Postgres backend the store handles deletion + the pgvector
    retriever queries fresh on every request — no rebuild needed.

    Args:
        memory_id: The id returned by `recall` or `list_memories`.
    """
    await _state.init()
    user_id = _current_user_id()
    removed = await _state.store.delete(memory_id, user_id=user_id)
    if not removed:
        return f"No memory found with id {memory_id!r}"

    # Rebuild path is only needed for in-memory retrievers (JSONL backend).
    # Postgres retrievers read the table on every request and pick up the
    # deletion automatically.
    if hasattr(_state.retriever, "dense") and hasattr(
        getattr(_state.retriever, "dense", None), "records",
    ):
        # JSONL path — rebuild from the (now-shorter) store.
        kept = await _state.store.load_all(user_id=user_id)
        dense = SemanticRetriever(embedder=_state.embedder)
        sparse = BM25Retriever()
        _state.retriever = HybridRetriever(dense=dense, sparse=sparse)
        await _state.retriever.ingest(kept)
        _state.pipeline.retriever = _state.retriever

    return f"Forgot memory {memory_id!r}"


@mcp.tool()
async def list_memories(
    entity: str | None = None,
    limit: int = 20,
) -> str:
    """Browse memories in the store.

    Args:
        entity: If set, only return memories where ``entity`` matches.
        limit: Max records to return. Default 20.
    """
    await _state.init()
    records = (await _state.store.load_all(user_id=_current_user_id()))
    if entity:
        records = [r for r in records if r.entity == entity]
    records = records[-limit:]
    serialized = [
        {
            "id": r.id,
            "content": r.content,
            "source": r.source,
            "entity": r.entity,
            "attribute": r.attribute,
            "value": r.value,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]
    return _format_memories(serialized) if serialized else "(no memories yet)"


@mcp.tool()
async def improve_query(text: str) -> str:
    """Run SAM's query-improvement heuristic to clarify a vague question.

    With the Ollama reasoner active, returns an LLM-rewritten query.
    Without it, returns the original text (heuristic SAM is passthrough).
    """
    await _state.init()
    if _state.sam.reasoner is None:
        return text
    classification = Classification(
        intent_type="question", entities=[], retrieval_hints={},
        confidence=0.5, source=ClassificationSource.KEYWORD_FALLBACK,
    )
    target = TargetDescriptor.for_claude()
    q = Pipeline.build_query(text, target, user_id=_current_user_id())
    result = await _state.sam.reason_from_scratch(q, classification)
    return result.improved_query or text


@mcp.tool()
async def status() -> str:
    """Report GML server state — pipeline wiring, memory count, paths."""
    await _state.init()
    n = len((await _state.store.load_all(user_id=_current_user_id())))
    return (
        f"GML memory server\n"
        f"  embedder:   {_state.embedder.version}\n"
        f"  retriever:  HybridRetriever (dense+BM25 RRF fusion)\n"
        f"  classifier: KeywordClassifier (regex)\n"
        f"  SAM:        {_state.sam_backend + ' ' + _state.sam_model_name if _state.sam_llm_enabled else 'heuristic-only (LLM unreachable)'}\n"
        f"  extractor:  {'MemoryExtractor (' + _state.sam_backend + ')' if _state.extractor_enabled else 'degraded — extract will return empty'}\n"
        f"  memories:   {n} in {getattr(_state.store, 'path', type(_state.store).__name__)}\n"
        f"  pipeline:   classify→embed→retrieve→[SAM | rerank+SAM]→assemble→translate\n"
    )


# ---------------------------------------------------------------------------
# Diagnostic tools — prove every stage fires and storage is consistent.
# ---------------------------------------------------------------------------


def _ms(t_start: float) -> int:
    return int((time.perf_counter() - t_start) * 1000)


# ---------------------------------------------------------------------------
# analyze(text) — rich intent classification on demand.
# Uses the same Ollama client as SAM. Zero-shot prompt, no fine-tuning.
# NOT wired into the pipeline — call it explicitly when you want depth.
# ---------------------------------------------------------------------------


_ANALYZE_PROMPT = """\
You are an intent analyzer for a memory-augmented AI assistant. Given the
user's message (and optionally the recent conversation history), produce a
structured analysis. Do NOT answer the user's question — only analyze it.

Conversation history (most recent first, may be empty):
{history}

Current user message:
{text!r}

Analyze and return JSON ONLY, no markdown fences, no prose. Schema:

{{
  "intent_type": "<one of: question, statement, instruction, request, clarification, greeting, vague_reference, sarcasm_negation, other>",
  "query_type": "<one of: factual_lookup, opinion, definition, comparison, status_check, action_request, social, ambiguous>",
  "tone": "<one of: neutral, formal, casual, sarcastic, frustrated, exploratory, urgent, playful>",
  "entities": [
    {{"text": "<surface form as written>", "type": "<person|service|product|tool|version|location|time|other>", "resolved": "<canonical name or null>"}}
  ],
  "references": [
    {{"phrase": "<the anaphor as written, e.g. 'that thing'>", "resolves_to": "<what it refers to, or null if unresolvable>"}}
  ],
  "is_followup": <true if this clearly continues a prior turn, else false>,
  "negation": <true if the message contains negation (no/not/never/don't/etc.) that flips meaning>,
  "confidence": <float 0..1 — how certain are you of the above>,
  "reasoning": "<one sentence explaining the most important signal you picked up>"
}}

Be decisive. If the message is short and clear, confidence should be high.
If it's ambiguous or relies heavily on missing context, lower the
confidence. Use empty arrays [] when fields don't apply.
"""


@mcp.tool()
async def analyze(text: str, history: str | None = None) -> str:
    """Rich zero-shot intent classification — on demand, NOT every turn.

    Returns a JSON analysis of the user's message:
      - intent_type   (question, statement, sarcasm_negation, vague_reference, …)
      - query_type    (factual_lookup, opinion, comparison, …)
      - tone          (neutral, sarcastic, frustrated, exploratory, …)
      - entities      (with type + canonical resolution)
      - references    (anaphora resolution: "that thing" → resolved target)
      - is_followup   (does this continue a prior turn?)
      - negation      (does the message flip meaning with no/not/never?)
      - confidence    (0..1)
      - reasoning     (one-sentence explanation)

    This is for inspection/debugging — pipeline does NOT call it per turn
    (would add 5-10s). Call it yourself when you want to understand how
    the system would interpret a specific message.

    Args:
        text: The user message to analyze.
        history: Optional short recent history ("user: ...\\nassistant: ...").
    """
    await _state.init()
    if _state.sam is None or _state.sam.reasoner is None or not _state.sam_llm_enabled:
        return (
            "analyze unavailable: Ollama LLM is not enabled. "
            "Start Ollama and pull a model (e.g. `ollama pull deepseek-r1:8b`)."
        )

    prompt = _ANALYZE_PROMPT.format(history=history or "(none)", text=text)
    client = _state.sam.reasoner.client
    try:
        gen = await client.generate(prompt, json_mode=True)
    except Exception as exc:
        return f"analyze failed: {type(exc).__name__}: {exc}"

    # Best-effort JSON extraction (Ollama with json_mode usually returns clean JSON)
    answer = gen.answer.strip()
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[-1] if "\n" in answer else answer[3:]
        if answer.endswith("```"):
            answer = answer.rsplit("```", 1)[0]
        answer = answer.strip()

    # Pretty-print for the calling AI to read easily
    import json as _json
    try:
        parsed = _json.loads(answer[answer.find("{"):answer.rfind("}") + 1])
        pretty = _json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return f"analyze: returned non-JSON\n---raw---\n{answer[:1500]}"

    return f"=== ANALYZE: {text!r} ===\n{pretty}"


@mcp.tool()
async def trace(text: str) -> str:
    """Run the full pipeline and return a STAGE-BY-STAGE breakdown.

    Use this to verify every component fires: Classifier → Embedder →
    Retriever.search (probe) → branch → (top50 → Reranker → SAM.resolve)
    or (SAM.reason_from_scratch) → Assembler → Translator. For each stage
    we show timing, input sizes, output sizes, and a sample of the data.

    This is what to run when you're asking "did the pipeline actually work
    end-to-end for THIS query?" — much more detailed than `query`.

    Args:
        text: The user's question, verbatim.
    """
    await _state.init()
    target = TargetDescriptor.for_claude()
    q = Pipeline.build_query(text=text, target=target, user_id=_current_user_id())
    pipe = _state.pipeline
    lines: list[str] = []
    lines.append(f"=== PIPELINE TRACE  query={text!r}  target={target.model_family.value} ===")

    # [1] Classifier
    t0 = time.perf_counter()
    classification = await pipe.classifier.classify(q)
    lines.append(f"\n[1] CLASSIFIER ({_ms(t0)}ms)  {type(pipe.classifier).__name__}")
    lines.append(f"    intent_type : {classification.intent_type}")
    lines.append(f"    entities    : {classification.entities}")
    lines.append(f"    confidence  : {classification.confidence:.2f}")
    lines.append(f"    source      : {classification.source.value}")
    lines.append(f"    degraded    : {classification.degraded}")

    # [2] Embedder
    t0 = time.perf_counter()
    embedded = await pipe.embedder.embed(q, classification)
    norm = math.sqrt(sum(x * x for x in embedded.vector))
    lines.append(f"\n[2] EMBEDDER ({_ms(t0)}ms)  {embedded.embedder_version}")
    lines.append(f"    vector dim  : {len(embedded.vector)}")
    lines.append(f"    L2 norm     : {norm:.4f}")
    lines.append(f"    first 6     : {[round(x, 4) for x in embedded.vector[:6]]}")
    lines.append(f"    last 3      : {[round(x, 4) for x in embedded.vector[-3:]]}")

    # [3] Retriever.search probe
    t0 = time.perf_counter()
    probe_hits = await pipe.retriever.search(embedded)
    lines.append(f"\n[3] RETRIEVER.search (probe, {_ms(t0)}ms)  {type(pipe.retriever).__name__}")
    lines.append(f"    hits        : {len(probe_hits)}")
    for i, h in enumerate(probe_hits[:3], 1):
        lines.append(f"    #{i} sim={h.similarity:.3f}  id={h.record.id}")
        lines.append(f"        {h.record.content[:100]!r}")

    # [4] Branch
    if not probe_hits:
        lines.append(f"\n[4] BRANCH: NO_MATCH → SAM.reason_from_scratch")
        t0 = time.perf_counter()
        resolved = await pipe.sam.reason_from_scratch(q, classification)
        lines.append(f"\n[5] SAM.reason_from_scratch ({_ms(t0)}ms)  reasoner={'LLM' if pipe.sam.reasoner else 'heuristic'}")
        ranked_in = 0
    else:
        lines.append(f"\n[4] BRANCH: YES_MATCH → top50 → rerank → SAM.resolve_conflicts")

        # Stage 4a: get_top_matches(50)
        t0 = time.perf_counter()
        top50 = await pipe.retriever.get_top_matches(
            embedded, k=pipe.config.retriever_top_k
        )
        lines.append(f"\n[5] RETRIEVER.get_top_matches ({_ms(t0)}ms)  k={pipe.config.retriever_top_k}")
        lines.append(f"    returned    : {len(top50)}")
        for i, h in enumerate(top50[:3], 1):
            lines.append(f"    #{i} sim={h.similarity:.3f}  id={h.record.id}")
            lines.append(f"        {h.record.content[:100]!r}")

        # Stage 4b: reranker.pick_best
        t0 = time.perf_counter()
        top10 = await pipe.reranker.pick_best(
            top50, q, k=pipe.config.reranker_top_k
        )
        lines.append(f"\n[6] RERANKER.pick_best ({_ms(t0)}ms)  {type(pipe.reranker).__name__}  k={pipe.config.reranker_top_k}")
        lines.append(f"    returned    : {len(top10)}")
        for i, h in enumerate(top10[:5], 1):
            lines.append(f"    #{i} final={h.final_score:.3f}  ({h.score_reason})")
            lines.append(f"        id={h.hit.record.id}  {h.hit.record.content[:90]!r}")

        # Stage 4c: sam.resolve_conflicts — with early-exit
        t0 = time.perf_counter()
        skip, reason = should_skip_sam(top10)
        if skip:
            resolved = ResolvedMemorySet(
                kept=top10,
                superseded=[],
                reason_from_scratch=False,
                notes=[f"SAM skipped: {reason}"],
            )
            lines.append(f"\n[7] SAM.resolve_conflicts ({_ms(t0)}ms)  SKIPPED — {reason}")
        else:
            resolved = await pipe.sam.resolve_conflicts(q, top10)
            lines.append(f"\n[7] SAM.resolve_conflicts ({_ms(t0)}ms)  reasoner={'LLM' if pipe.sam.reasoner else 'heuristic'}  (decision: {reason})")
        ranked_in = len(top10)

    lines.append(f"    kept        : {len(resolved.kept)}")
    lines.append(f"    superseded  : {len(resolved.superseded)} pair(s)")
    if resolved.superseded[:3]:
        for loser, winner in resolved.superseded[:3]:
            lines.append(f"      {loser} ⇠ {winner}")
    lines.append(f"    reason_from_scratch: {resolved.reason_from_scratch}")
    lines.append(f"    improved_query     : {resolved.improved_query!r}")
    if resolved.reasoning_content:
        snippet = resolved.reasoning_content.replace("\n", " ")[:160]
        lines.append(f"    reasoning_content  : {snippet!r}...")

    # Stage 5: Assembler
    tokenizer, assembler, template_overhead = pipe._resolve_target(target)
    t0 = time.perf_counter()
    context = assembler.package(
        resolved, q, template_overhead_tokens=template_overhead,
        final=pipe.config.assembler_final_k,
    )
    lines.append(f"\n[8] ASSEMBLER.package ({_ms(t0)}ms)  {type(assembler).__name__}  final_k={pipe.config.assembler_final_k}")
    lines.append(f"    selected    : {len(context.selected)}")
    lines.append(f"    dropped     : {len(context.dropped_ids)} (budget)")
    used = context.budget_total - context.budget_remaining
    lines.append(f"    budget used : {used}/{context.budget_total} tokens ({used*100//max(context.budget_total,1)}%)")
    for i, h in enumerate(context.selected[:3], 1):
        lines.append(f"    #{i} id={h.hit.record.id}  {h.hit.record.content[:90]!r}")

    # Stage 6: Translator
    t0 = time.perf_counter()
    payload = pipe.translator.translate(context, config_hash=pipe._config_hash)
    lines.append(f"\n[9] TRANSLATOR ({_ms(t0)}ms)  target={payload.target.model_family.value}")
    lines.append(f"    formatted_context: {len(payload.formatted_context)} chars")
    lines.append(f"    user_query       : {payload.user_query!r}")
    lines.append(f"    items_included   : {payload.metadata.get('items_included')}")
    lines.append(f"    query_was_improved: {payload.metadata.get('query_was_improved')}")

    lines.append(f"\n=== END TRACE ===")
    lines.append(f"--- formatted_context (what the target AI receives) ---")
    lines.append(payload.formatted_context)
    return "\n".join(lines)


@mcp.tool()
async def diag() -> str:
    """Storage + retriever diagnostic.

    Prove that (a) memories are persisted to disk, (b) the same set is
    loaded into the retriever's index, (c) embeddings are real vectors
    (shows dim and a sample for one record). Use this to confirm the
    embeddings-are-stored-properly question without running a query.
    """
    await _state.init()
    n_disk = len((await _state.store.load_all(user_id=_current_user_id())))
    dense = _state.retriever.dense  # SemanticRetriever
    n_index = len(dense.records)
    n_vectors = len(dense._vectors)

    lines = ["=== STORAGE + INDEX DIAGNOSTIC ==="]
    lines.append(f"  memory store  : {getattr(_state.store, 'path', type(_state.store).__name__)}")
    lines.append(f"  records on disk : {n_disk}")
    lines.append(f"  records in dense index : {n_index}")
    lines.append(f"  vectors in dense index : {n_vectors}")
    lines.append(f"  consistency   : {'OK' if n_disk <= n_index and n_index == n_vectors else 'MISMATCH'}")
    lines.append(f"  embedder      : {_state.embedder.version}")

    if dense.records:
        sample = dense.records[-1]  # most recently ingested
        vec = dense._vectors.get(sample.id, [])
        norm = math.sqrt(sum(x * x for x in vec)) if vec else 0.0
        lines.append(f"\n--- Sample (most recent record) ---")
        lines.append(f"  id      : {sample.id}")
        lines.append(f"  source  : {sample.source}")
        lines.append(f"  content : {sample.content!r}")
        lines.append(f"  vector  : dim={len(vec)}  L2_norm={norm:.4f}")
        lines.append(f"  first 6 : {[round(x, 4) for x in vec[:6]]}")
        lines.append(f"  last 3  : {[round(x, 4) for x in vec[-3:]]}")

    # Check sparse (BM25) too
    sparse = _state.retriever.sparse
    sparse_n = len(getattr(sparse, "_records", []) or getattr(sparse, "records", []) or [])
    lines.append(f"\n--- BM25 (sparse) index ---")
    lines.append(f"  records : {sparse_n}")

    return "\n".join(lines)


def run() -> None:
    """Entry point invoked by ``gml mcp`` and by ``python -m orchestration.mcp_server``.

    Transport is selected by ``GML_MCP_TRANSPORT``:

      * ``stdio`` (default) — for Claude Desktop, Cursor, Windsurf, Antigravity
        when running locally. Stdout is JSON-RPC; we route our own logs to stderr.
      * ``streamable-http`` — modern HTTP transport. Use this when hosting the
        MCP server behind nginx on a public box (Claude Desktop, Cursor, etc.
        all support remote MCP servers via streamable-http). Auth is layered
        via Starlette middleware that consults the same ``users.jsonl`` the
        FastAPI server uses, so the same per-user keys work here.
      * ``sse`` — legacy SSE transport. Don't pick this for new deployments;
        kept for backward compatibility.
    """
    transport = os.environ.get("GML_MCP_TRANSPORT", "stdio").strip().lower()

    # Proxy mode: forward tool calls to the running HTTP API instead of
    # loading the embedder/rerankers in-process (~1.5 GB per child behind
    # the relay connector). See orchestration.mcp_proxy.
    if transport == "stdio" and os.environ.get("GML_MCP_PROXY_URL", "").strip():
        from orchestration.mcp_proxy import run as proxy_run
        proxy_run()
        return

    if transport == "stdio":
        # Stdout is JSON-RPC; route our own logs to stderr so we don't corrupt it.
        _set_log_stream(sys.stderr)
        sys.stderr.write("gml-memory MCP server ready (stdio)\n")
        mcp.run(transport="stdio")
        return

    if transport in ("streamable-http", "http", "sse"):
        # Wrap the FastMCP-built Starlette app with our auth middleware so
        # the same user keys gate the MCP surface. The master key works too
        # (admins want to test). If neither is configured, we run open with
        # a stderr warning — same policy as the FastAPI server.
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from orchestration.storage import _is_postgres, make_user_key_store

        master_key = os.environ.get("GML_API_KEY", "").strip()
        # Auth is on whenever a master key is set OR the storage backend is
        # postgres (production should never be open). The user store is
        # resolved lazily on first request (same pattern as the FastAPI side)
        # so we don't hit the DB at startup.
        auth_enabled = bool(master_key) or _is_postgres()
        _user_store_cache: dict = {"instance": None}

        async def _store():
            if _user_store_cache["instance"] is None:
                _user_store_cache["instance"] = await make_user_key_store()
            return _user_store_cache["instance"]

        if not auth_enabled:
            sys.stderr.write(
                "[gml-mcp] WARNING: GML_API_KEY unset and storage=jsonl — "
                "MCP HTTP transport is OPEN. Set GML_API_KEY or switch to "
                "GML_STORAGE_BACKEND=postgres before exposing publicly.\n"
            )

        class _AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if not auth_enabled:
                    # Even in open mode, set the contextvar to None so tools
                    # have a consistent way to read it.
                    token = _CURRENT_USER_ID.set(None)
                    try:
                        return await call_next(request)
                    finally:
                        _CURRENT_USER_ID.reset(token)
                if request.method == "OPTIONS":
                    return await call_next(request)
                if request.url.path in ("/health", "/healthz"):
                    return await call_next(request)
                provided = request.headers.get("x-api-key") or ""
                if not provided:
                    auth = request.headers.get("authorization") or ""
                    if auth.lower().startswith("bearer "):
                        provided = auth.split(None, 1)[1].strip()
                is_master = bool(master_key) and provided == master_key
                rec = None
                if not is_master and provided:
                    store = await _store()
                    rec = await store.lookup(provided)
                if not is_master and rec is None:
                    return JSONResponse({"detail": "unauthorized"}, status_code=401)
                resolved_user = "admin" if is_master else rec.user_id
                request.state.user_id = resolved_user
                request.state.is_master = is_master
                # Push the user_id into the ContextVar so MCP tool functions
                # (which can't accept a Request directly) can read it.
                token = _CURRENT_USER_ID.set(
                    None if is_master else resolved_user
                )
                try:
                    return await call_next(request)
                finally:
                    _CURRENT_USER_ID.reset(token)

        # FastMCP gives us the underlying Starlette app for either transport.
        if transport == "sse":
            app = mcp.sse_app()
            url_path = os.environ.get("GML_MCP_PATH", mcp.settings.sse_path)
        else:
            app = mcp.streamable_http_app()
            url_path = os.environ.get("GML_MCP_PATH", mcp.settings.streamable_http_path)

        # Add the auth middleware. Starlette apps accept add_middleware()
        # after construction.
        app.add_middleware(_AuthMiddleware)

        host = os.environ.get("GML_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("GML_MCP_PORT", "8765"))
        sys.stderr.write(
            f"[gml-mcp] streamable-http transport on http://{host}:{port}{url_path}\n"
            f"  auth: {'enabled' if auth_enabled else 'DISABLED (dev only)'}\n"
        )
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="info")
        return

    raise SystemExit(
        f"Unknown GML_MCP_TRANSPORT={transport!r}. "
        "Choices: stdio, streamable-http, sse."
    )


if __name__ == "__main__":
    run()
