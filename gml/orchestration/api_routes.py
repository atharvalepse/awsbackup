"""``/api`` HTTP surface for the GML web UI.

This router is the stable contract the Next.js frontend consumes. Every route
reuses existing orchestration logic — it adds no new memory/pipeline
behaviour, only an HTTP shape over what the MCP tools and pipeline already do,
plus a derived cluster/graph view (see :mod:`orchestration.graph_projection`).

Schema mapping (backend → API):
  * ``importance`` ← ``MemoryItem.authority_score`` (SDP maps its importance
    score here; LLM-extracted memories carry the extractor's default).
  * ``confidence`` ← ``raw_metadata["confidence"]`` when present (SDP), else
    falls back to ``authority_score``.
  * ``cluster_id`` ← KMeans label from the graph projector (derived, cached).

Registered onto the FastAPI app in :func:`orchestration.server.build_app` via
``app.include_router(build_api_router(state, run_trace=...))``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import base64
import io
import json
import os
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

from orchestration import __version__
from orchestration.connectors import codex as codex_connector
from orchestration.graph_projection import GraphProjector
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline import MemoryItem, Pipeline
from orchestration.ingestion.summarizer import summarize_turn
from orchestration.pipeline.contracts import Classification, ClassificationSource
from orchestration.retriever.query_expansion import (
    expand_query,
    query_expansion_enabled,
    rrf_merge,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orchestration.server import ServerState

slog = StructuredLogger("api")

# A run_trace callable: (state, text, user_id=...) -> structured trace dict.
RunTrace = Callable[..., Awaitable[dict]]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ApiMemory(BaseModel):
    id: str
    content: str
    entity: str | None = None
    # canonical entity id from write-time entity resolution (migration 014);
    # claims about "GML" and "Gigzs Multi-LLM Layer" share one entity_id
    entity_id: str | None = None
    attribute: str | None = None
    value: str | None = None
    confidence: float
    importance: float
    cluster_id: int | None = None
    source: str
    pinned: bool = False
    timestamp: str
    summary_short: str | None = None
    # Bitemporal audit surface (migration 013): validity interval, current-
    # belief flag, and the forward supersession link. None/True for rows
    # from stores without bitemporal data.
    valid_from: str | None = None
    valid_to: str | None = None
    is_latest: bool = True
    superseded_by: str | None = None


class MemoryListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    memories: list[ApiMemory]


class GateDecision(BaseModel):
    decision: str
    nearest_id: str | None = None
    similarity: float | None = None
    reason: str | None = None
    matched_signals: list[str] = []
    created_at: str


class LineageResponse(BaseModel):
    """Answer to 'where did this memory come from and what happened to it':
    validity interval, supersession links both ways, conflict partners, and
    the write-gate decision ledger."""

    id: str
    valid_from: str | None = None
    valid_to: str | None = None
    is_latest: bool = True
    session_id: str | None = None
    superseded_by: str | None = None
    supersedes: list[str] = []
    conflict_with: list[str] = []
    gate_decisions: list[GateDecision] = []


class Relationship(BaseModel):
    memory_id: str
    entity: str | None = None
    value: str | None = None
    cluster_id: int | None = None
    kind: str  # "similarity" | "entity"
    weight: float | None = None


class MemoryDetailResponse(BaseModel):
    memory: ApiMemory
    relationships: list[Relationship]


class GraphNode(BaseModel):
    id: str
    label: str
    entity: str | None = None
    value: str | None = None
    cluster_id: int
    importance: float
    val: float
    x: float
    y: float


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    depth: int


class Cluster(BaseModel):
    id: int
    label: str
    centroid: dict
    size: int
    color_hint: str


class ClusterListResponse(BaseModel):
    clusters: list[Cluster]


# Candidate-pool size fetched before reranking on /memory/recall. Matches the
# pipeline's retriever_top_k=50 so the cross-encoder has enough to work with.
_RECALL_POOL = 50


class RecallRequest(BaseModel):
    # Accept either "query" or "text" so the recall/trace/chat surfaces are
    # consistent and clients can't get tripped by the field name.
    query: str = Field(min_length=1, validation_alias=AliasChoices("query", "text"))
    top_k: int = Field(5, ge=1, le=50)
    # Rerank the candidate pool with the cross-encoder before returning
    # (default). Without this, results are raw RRF order — the top hit is fine
    # but lower slots fill with unrelated memories at deceptive rank-decay
    # scores. Set False only for latency-sensitive live search where ordering
    # matters less than speed.
    rerank: bool = True
    # Allow query expansion (paraphrase -> multi-retrieve -> RRF merge) when the
    # server has it enabled (GML_QUERY_EXPANSION). Set False to force the
    # single-query path even there — e.g. the latency-sensitive live search.
    expand: bool = True
    # Time travel: return the belief state AS OF this instant instead of
    # current beliefs. Pydantic parses/validates it into a real datetime —
    # it never reaches SQL as text (parameterized in the retrievers).
    as_of: datetime | None = None


class RecallResult(BaseModel):
    memory: ApiMemory
    score: float
    why: str | None = None
    # True when this claim has unresolved contradicting active claims
    # (write-gate conflict_with links). conflicting_ids lists them so a
    # client can fetch/show both sides — we never silently pick a winner.
    conflict: bool = False
    conflicting_ids: list[str] = []


class RecallResponse(BaseModel):
    query: str
    results: list[RecallResult]


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1)
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    source: str = "manual"
    authority_score: float = Field(0.8, ge=0.0, le=1.0)
    pinned: bool = False


class IngestRequest(BaseModel):
    user_query: str
    assistant_reply: str


class IngestResponse(BaseModel):
    mode: str  # "llm" | "sdp"
    count: int
    created: list[str]
    detail: str | None = None
    job_id: str | None = None


class IngestJobResponse(BaseModel):
    job_id: str
    state: str  # "queued" | "running" | "done" | "failed"
    facts_added: int = 0
    last_error: str | None = None


class ConversationFact(BaseModel):
    id: str | None = None
    content: str
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    confidence: float | None = None


class ApiConversation(BaseModel):
    id: str
    title: str | None = None
    summary: str | None = None
    user_prompt: str | None = None
    ai_response: str | None = None
    source_url: str | None = None
    source_model: str | None = None
    facts: list[ConversationFact] = []
    fact_count: int = 0
    created_at: str | None = None


class ConversationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    conversations: list[ApiConversation]


class ConversationCreateRequest(BaseModel):
    # Accept both snake_case (gmlcore-native) and the extension's camelCase.
    model_config = {"populate_by_name": True}
    user_prompt: str = Field("", alias="userPrompt")
    ai_response: str = Field("", alias="aiResponse")
    source: str | None = Field(None, alias="source")
    model: str | None = Field(None, alias="model")


class ConversationCreateResponse(BaseModel):
    id: str
    title: str | None = None
    summary: str | None = None
    fact_count: int = 0
    job_id: str | None = None


class TraceRequest(BaseModel):
    text: str = Field(min_length=1, validation_alias=AliasChoices("text", "query"))


class SynthesizeResponse(BaseModel):
    query: str
    context: str
    items_included: int


# ---------------------------------------------------------------------------
# Schema mappers
# ---------------------------------------------------------------------------


def _confidence(item: MemoryItem) -> float:
    raw = item.raw_metadata.get("confidence")
    if isinstance(raw, (int, float)):
        return float(raw)
    return item.authority_score


def _node_label(item: MemoryItem) -> str:
    if item.entity and item.value:
        return f"{item.entity}: {item.value[:32]}"
    if item.entity:
        return item.entity
    return item.content[:40]


def to_api_memory(item: MemoryItem, cluster_id: int | None) -> ApiMemory:
    superseded_by = (item.raw_metadata or {}).get("superseded_by")
    return ApiMemory(
        id=item.id,
        content=item.content,
        entity=item.entity,
        entity_id=item.entity_id,
        attribute=item.attribute,
        value=item.value,
        confidence=_confidence(item),
        importance=item.authority_score,
        cluster_id=cluster_id,
        source=item.source,
        pinned=item.pinned,
        timestamp=item.timestamp.isoformat(),
        summary_short=item.summary_short,
        valid_from=item.valid_from.isoformat() if item.valid_from else None,
        valid_to=item.valid_to.isoformat() if item.valid_to else None,
        is_latest=item.is_latest,
        superseded_by=superseded_by if isinstance(superseded_by, str) else None,
    )


def _why_matched(item: MemoryItem, query: str) -> str | None:
    """Cheap, honest 'why this matched' — token overlap on the entity.

    Real SAM reasoning is only produced on the full pipeline path
    (/api/memory/synthesize, /api/memory/trace), not on raw recall, so we
    surface a derived hint rather than claim reasoning we didn't run.
    """
    if not item.entity:
        return None
    q = query.lower()
    if item.entity.lower() in q or any(
        tok in q for tok in item.entity.lower().split() if len(tok) >= 4
    ):
        return f"matches entity “{item.entity}”"
    return None


def _sse(event: str, data: dict) -> bytes:
    """Encode one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def build_api_router(state: "ServerState", run_trace: RunTrace) -> APIRouter:
    """Build the ``/api`` router bound to a :class:`ServerState`.

    ``run_trace`` is injected (rather than imported) to avoid a circular
    import with :mod:`orchestration.server`, which owns the trace logic.
    """
    router = APIRouter(prefix="/api", tags=["web"])

    # GraphProjector pulls from an in-memory retriever .records list. That
    # only exists in JSONL mode; for Postgres we won't build it (graph
    # endpoint degrades to an empty result and a soft note). When the
    # in-memory cache IS available, the projector caches its PCA + KMeans
    # behind .get() and refreshes when len(records) changes.
    # Built lazily on first use, not at router-construction time: under the
    # make_app() factory the state is a deferred proxy and state.retriever
    # isn't available yet. GraphProjector only applies to the in-memory (JSONL)
    # retriever that exposes .records; Postgres mode stays None (graph degrades).
    _proj_cache: dict = {"built": False, "proj": None}

    def _projector() -> GraphProjector | None:
        if not _proj_cache["built"]:
            _proj_cache["built"] = True
            if hasattr(state.retriever, "records"):
                _proj_cache["proj"] = GraphProjector(state.retriever)
        return _proj_cache["proj"]

    def _projection():
        """Computed projection (PCA + KMeans) or None on the Postgres backend."""
        p = _projector()
        return p.get() if p else None

    async def _projection_for(user_id: str | None):
        """Per-request projection for either backend (PCA + KMeans + kNN).

        JSONL: the memoized in-memory projector (single tenant). Postgres:
        build one from this user's stored vectors (RLS-scoped), so the graph
        is per-tenant. Returns None if there's nothing to project.
        """
        p = _projector()
        if p is not None:
            return p.get()
        store = state.memory_store
        if hasattr(store, "load_with_vectors"):
            items, vectors = await store.load_with_vectors(user_id)
            if items and vectors:
                return GraphProjector.for_items(items, vectors).get()
        return None

    async def _load_user_memories(user_id: str | None) -> list[MemoryItem]:
        """Backend-agnostic read of one user's memories.

        Postgres backend: scoped via RLS using the user_id we set on the
        session var inside the storage adapter.
        JSONL backend: returns everything (single tenant). ``user_id`` is
        ignored.
        """
        return await state.memory_store.load_all(user_id=user_id)

    # -- health ------------------------------------------------------------

    @router.get("/health")
    async def api_health() -> dict:
        # Total memory count: JSONL backend reads the in-memory retriever
        # cache (fast); Postgres backend reads via SELECT count(*). We do
        # the cheaper one available.
        if hasattr(state.retriever, "records"):
            memories = len(state.retriever.records)
        else:
            memories = len(await state.memory_store.load_all())
        return {
            "status": "ok",
            "version": __version__,
            "memories": memories,
            "embedder": state.embedder.version,
        }

    # -- memories ----------------------------------------------------------

    @router.get("/memories", response_model=MemoryListResponse)
    async def list_memories(
        request: Request,
        cluster: int | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> MemoryListResponse:
        items = await _load_user_memories(_user_id_from(request))
        proj = _projection()
        if cluster is not None and proj:
            items = [r for r in items if proj.cluster_by_id.get(r.id) == cluster]
        total = len(items)
        page = items[offset : offset + limit]
        return MemoryListResponse(
            total=total,
            limit=limit,
            offset=offset,
            memories=[
                to_api_memory(r, proj.cluster_by_id.get(r.id) if proj else None)
                for r in page
            ],
        )

    @router.get("/memories/graph", response_model=GraphResponse)
    async def memory_graph(
        request: Request, depth: int = Query(2, ge=1, le=5),
    ) -> GraphResponse:
        # Graph projection needs vectors. In JSONL mode they live in the
        # SemanticRetriever's in-memory cache; in Postgres mode they live in
        # the DB and we project them per-user (RLS-scoped) on demand. Either
        # way _projection_for returns a Projection or None (empty graph).
        uid = _user_id_from(request)
        proj = await _projection_for(uid)
        if proj is None:
            return GraphResponse(nodes=[], edges=[], depth=depth)
        nodes: list[GraphNode] = []
        for r in await _load_user_memories(uid):
            if r.id not in proj.coords_by_id:
                continue  # not embedded → not in the projection
            x, y = proj.coords_by_id[r.id]
            imp = r.authority_score
            nodes.append(
                GraphNode(
                    id=r.id,
                    label=_node_label(r),
                    entity=r.entity,
                    value=r.value,
                    cluster_id=proj.cluster_by_id.get(r.id, 0),
                    importance=imp,
                    val=(max(imp, 0.0) ** 0.5) * 4 + 2,
                    x=x,
                    y=y,
                )
            )
        edges = [GraphEdge(**e) for e in proj.edges]
        return GraphResponse(nodes=nodes, edges=edges, depth=depth)

    @router.get("/memories/{memory_id}", response_model=MemoryDetailResponse)
    async def get_memory(memory_id: str, request: Request) -> MemoryDetailResponse:
        items = await _load_user_memories(_user_id_from(request))
        rec = next((r for r in items if r.id == memory_id), None)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"memory {memory_id!r} not found")
        rec_by_id = {r.id: r for r in items}

        proj = _projection()
        rels: dict[str, Relationship] = {}
        if proj:
            for e in proj.edges:
                other = None
                if e["source"] == memory_id:
                    other = e["target"]
                elif e["target"] == memory_id:
                    other = e["source"]
                if other and other in rec_by_id:
                    o = rec_by_id[other]
                    rels[other] = Relationship(
                        memory_id=other,
                        entity=o.entity,
                        value=o.value,
                        cluster_id=proj.cluster_by_id.get(other),
                        kind="similarity",
                        weight=e["weight"],
                    )
        # Entity-shared relationships work in both backends — pure metadata join.
        if rec.entity:
            for r in items:
                if r.id != memory_id and r.entity == rec.entity and r.id not in rels:
                    rels[r.id] = Relationship(
                        memory_id=r.id,
                        entity=r.entity,
                        value=r.value,
                        cluster_id=proj.cluster_by_id.get(r.id) if proj else None,
                        kind="entity",
                    )
        return MemoryDetailResponse(
            memory=to_api_memory(rec, proj.cluster_by_id.get(memory_id) if proj else None),
            relationships=list(rels.values()),
        )

    @router.get("/memories/{memory_id}/lineage", response_model=LineageResponse)
    async def memory_lineage(memory_id: str, request: Request) -> LineageResponse:
        uid = _user_id_from(request)
        store = state.memory_store
        if hasattr(store, "get_lineage"):
            data = await store.get_lineage(memory_id, user_id=uid)
            if data is None:
                raise HTTPException(
                    status_code=404, detail=f"memory {memory_id!r} not found"
                )
            return LineageResponse(**data)
        # JSONL backend: metadata-only lineage (no gate ledger).
        items = await _load_user_memories(uid)
        rec = next((r for r in items if r.id == memory_id), None)
        if rec is None:
            raise HTTPException(
                status_code=404, detail=f"memory {memory_id!r} not found"
            )
        meta = rec.raw_metadata or {}
        forward = meta.get("supersedes")
        superseded_by = meta.get("superseded_by")
        return LineageResponse(
            id=rec.id,
            valid_from=rec.valid_from.isoformat() if rec.valid_from else None,
            valid_to=rec.valid_to.isoformat() if rec.valid_to else None,
            is_latest=rec.is_latest,
            session_id=meta.get("session_id"),
            superseded_by=superseded_by if isinstance(superseded_by, str) else None,
            supersedes=[forward] if isinstance(forward, str) else [],
            conflict_with=sorted(meta.get("conflict_with") or []),
        )

    @router.post("/memories", response_model=ApiMemory, status_code=201)
    async def create_memory(req: MemoryCreateRequest, request: Request) -> ApiMemory:
        """Persist one explicit memory, scoped to the authenticated user.

        The user-scoped counterpart of the legacy unscoped ``POST /memories``;
        the MCP proxy's `remember` tool lands here.
        """
        user_id = _user_id_from(request)
        item = MemoryItem(
            id=f"manual-{uuid.uuid4().hex[:12]}",
            content=req.content,
            entity=req.entity,
            attribute=req.attribute,
            value=req.value,
            timestamp=datetime.now(timezone.utc),
            source=req.source,
            authority_score=req.authority_score,
            pinned=req.pinned,
        )
        if user_id and hasattr(state.memory_store, "check_quota_or_raise"):
            new_bytes = len(item.content.encode("utf-8"))
            ok, qstatus = await state.memory_store.check_quota_or_raise(
                user_id, new_bytes,
            )
            if not ok:
                raise HTTPException(
                    status_code=402,
                    detail={
                        "error": "storage_limit_exceeded",
                        "used_bytes": qstatus.get("bytes_used"),
                        "quota_bytes": qstatus.get("quota_bytes"),
                    },
                )
        async with state._lock:
            await state.memory_store.add(item, user_id=user_id)
            await state.retriever.ingest([item])
        slog.info(event="memory_added", memory_id=item.id,
                  source=item.source, user_id=user_id)
        return to_api_memory(item, None)

    @router.delete("/memories/{memory_id}")
    async def forget(memory_id: str, request: Request) -> dict:
        async with state._lock:
            removed = await state.memory_store.delete(
                memory_id, user_id=_user_id_from(request),
            )
            if not removed:
                raise HTTPException(
                    status_code=404, detail=f"memory {memory_id!r} not found"
                )
            # In-memory retrievers (JSONL backend) maintain their own index;
            # remove there too. Postgres-backed retrievers have no in-memory
            # mirror to update — the next query reads the (now-deleted) row
            # straight from the DB.
            if hasattr(state.retriever, "remove"):
                state.retriever.remove([memory_id])
        slog.info(
            event="memory_forgotten",
            memory_id=memory_id, user_id=_user_id_from(request),
        )
        return {"deleted": memory_id}

    # -- clusters ----------------------------------------------------------

    @router.get("/clusters", response_model=ClusterListResponse)
    async def clusters(request: Request) -> ClusterListResponse:
        proj = await _projection_for(_user_id_from(request))
        if proj is None:
            return ClusterListResponse(clusters=[])
        return ClusterListResponse(clusters=[Cluster(**c) for c in proj.clusters])

    # -- recall / synthesize / trace --------------------------------------

    # Helper — pull the authenticated user_id off request.state. The auth
    # middleware (server.py) sets it on every authenticated request. Master
    # admin sees "admin"; we treat that as None (unscoped) for query routing.
    def _user_id_from(request: Request) -> str | None:
        uid = getattr(request.state, "user_id", None)
        return uid if uid and uid != "admin" else None

    @router.post("/memory/recall", response_model=RecallResponse)
    async def recall(req: RecallRequest, request: Request) -> RecallResponse:
        classification = Classification(
            intent_type="question",
            entities=[],
            retrieval_hints={},
            confidence=0.5,
            source=ClassificationSource.KEYWORD_FALLBACK,
        )
        q = Pipeline.build_query(
            req.query, state.default_target,
            user_id=_user_id_from(request),
            as_of=req.as_of,
        )
        proj = _projection()
        reranker = state.pipeline.reranker  # always set on the pipeline
        do_rerank = req.rerank and reranker is not None
        # Reranking trims a wide pool down to top_k; without it we retrieve
        # exactly top_k. The cosine junk-floor lives at the dense retriever
        # (match_threshold) — never on the fused RRF score.
        pool_k = _RECALL_POOL if do_rerank else req.top_k

        # Build the candidate pool. With query expansion enabled, paraphrase
        # the query, retrieve for each phrasing concurrently, and RRF-merge the
        # result sets so memories that several phrasings agree on rise. Falls
        # back to the single-query path if expansion is off or yields nothing.
        uid = _user_id_from(request)
        variants = [req.query]
        if req.expand and query_expansion_enabled():
            variants += await expand_query(req.query, n=3)
        if len(variants) > 1:
            embeddeds = await asyncio.gather(*[
                state.embedder.embed(
                    Pipeline.build_query(
                        v, state.default_target, user_id=uid, as_of=req.as_of,
                    ),
                    classification,
                )
                for v in variants
            ])
            result_sets = await asyncio.gather(*[
                state.retriever.get_top_matches(emb, k=pool_k) for emb in embeddeds
            ])
            pool = rrf_merge(result_sets, k=pool_k)
        else:
            embedded = await state.embedder.embed(q, classification)
            pool = await state.retriever.get_top_matches(embedded, k=pool_k)

        # Collapse chunks of the same long memory to their best-scoring chunk so
        # one source can't occupy several slots. pool is ordered best-first, so
        # keeping the first per group keeps the best. Atomic facts (no parent)
        # group by their own id, i.e. are unaffected.
        seen_groups: set[str] = set()
        deduped = []
        for h in pool:
            group = h.record.parent_memory_id or h.record.id
            if group in seen_groups:
                continue
            seen_groups.add(group)
            deduped.append(h)
        pool = deduped

        def _result(record, score: float) -> RecallResult:
            conflicting = sorted(
                (record.raw_metadata or {}).get("conflict_with") or []
            )
            why = _why_matched(record, req.query)
            if conflicting:
                note = f"contradicts {len(conflicting)} other active claim(s)"
                why = f"{why}; {note}" if why else note
            return RecallResult(
                memory=to_api_memory(
                    record, proj.cluster_by_id.get(record.id) if proj else None
                ),
                score=round(score, 4),
                why=why,
                conflict=bool(conflicting),
                conflicting_ids=conflicting,
            )

        if do_rerank:
            ranked = await reranker.pick_best(pool, q, k=req.top_k)
            results = [_result(r.record, r.final_score) for r in ranked]
        else:
            results = [_result(h.record, h.similarity) for h in pool[:req.top_k]]
        return RecallResponse(query=req.query, results=results)

    @router.get("/memory/synthesize", response_model=SynthesizeResponse)
    async def synthesize(
        request: Request,
        query: str = Query(min_length=1),
        as_of: datetime | None = Query(
            None, description="time-travel: belief state as of this instant"
        ),
    ) -> SynthesizeResponse:
        q = Pipeline.build_query(
            query, state.default_target,
            user_id=_user_id_from(request),
            as_of=as_of,
        )
        payload = await state.pipeline.run(q)
        return SynthesizeResponse(
            query=query,
            context=payload.formatted_context,
            items_included=int(payload.metadata.get("items_included", 0) or 0),
        )

    @router.post("/memory/trace")
    async def trace(req: TraceRequest, request: Request) -> dict:
        return await run_trace(state, req.text, user_id=_user_id_from(request))

    @router.post("/memory/recall/stream")
    async def recall_stream(req: RecallRequest, request: Request) -> StreamingResponse:
        """Full-pipeline recall with live per-stage progress over SSE.

        Emits ``event: stage`` as each pipeline stage completes (drives the
        UI's 7-dot indicator from real timing), then ``event: done`` with the
        reranked results + SAM reasoning. Distinct from ``/memory/recall``,
        which is the fast raw-retrieval path used for live search.
        """
        from orchestration.server import stream_pipeline_trace
        user_id = _user_id_from(request)

        async def gen():
            proj = _projection()
            # Build id→record map from the user's memories. Postgres mode
            # uses load_all (scoped via RLS); JSONL falls through to the
            # same call which is just slower but correct.
            items = await state.memory_store.load_all(user_id=user_id)
            rec_by_id = {r.id: r for r in items}
            try:
                async for kind, payload in stream_pipeline_trace(
                    state, req.query, user_id=user_id,
                ):
                    if kind == "stage":
                        yield _sse("stage", {
                            "stage": payload["stage"],
                            "duration_ms": payload["duration_ms"],
                        })
                    elif kind == "done":
                        ann = payload["annotations"]
                        results = []
                        for r in ann["ranked"][: req.top_k]:
                            rec = rec_by_id.get(r["id"])
                            if rec is None:
                                continue
                            results.append(
                                RecallResult(
                                    memory=to_api_memory(rec, proj.cluster_by_id.get(rec.id) if proj else None),
                                    score=round(r["final_score"], 4),
                                    why=_why_matched(rec, req.query),
                                ).model_dump()
                            )
                        yield _sse("done", {
                            "results": results,
                            "improved_query": ann.get("improved_query"),
                            "sam_reasoning": ann.get("sam_reasoning"),
                            "formatted_context": payload["formatted_context"],
                        })
            except Exception as exc:  # surface failures as an SSE error event
                yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    @router.post("/memory/trace/stream")
    async def trace_stream(req: TraceRequest, request: Request) -> StreamingResponse:
        """Same trace as POST /memory/trace, but streamed stage-by-stage (SSE)."""
        from orchestration.server import stream_pipeline_trace
        user_id = _user_id_from(request)

        async def gen():
            try:
                async for kind, payload in stream_pipeline_trace(
                    state, req.text, user_id=user_id,
                ):
                    if kind == "stage":
                        yield _sse("stage", {
                            "stage": payload["stage"],
                            "duration_ms": payload["duration_ms"],
                        })
                    elif kind == "done":
                        yield _sse("done", payload)
            except Exception as exc:
                yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    # -- ingest ------------------------------------------------------------
    # Both endpoints route through AALConverter so every persisted row gets
    # canonical {simplemem, sjson} columns populated. Reduces drift between
    # the two paths and lets retrievers/SAM read either view.

    from orchestration.aal import AALConverter
    _aal_converter = AALConverter()

    # Background LLM-extraction tasks. The local model on CPU is slow (tens of
    # seconds), so we never make the client wait on it — extraction runs as a
    # fire-and-forget task and the memories show up on the next dashboard /
    # extension refresh. Keep strong refs so tasks aren't GC'd mid-flight.
    _ingest_tasks: set = set()
    # Job registry for async LLM extraction so clients poll real state
    # instead of inferring it from /api/memories growth. In-memory (single
    # process, same as _ingest_tasks); FIFO-capped to bound growth.
    _ingest_jobs: "dict[str, dict]" = {}
    _INGEST_JOBS_MAX = 500

    def _job_set(job_id: str, **fields) -> None:
        rec = _ingest_jobs.get(job_id)
        if rec is None:
            if len(_ingest_jobs) >= _INGEST_JOBS_MAX:
                for k in list(_ingest_jobs.keys())[: max(1, _INGEST_JOBS_MAX // 10)]:
                    _ingest_jobs.pop(k, None)
            rec = {"job_id": job_id, "state": "queued",
                   "facts_added": 0, "last_error": None}
            _ingest_jobs[job_id] = rec
        rec.update(fields)

    async def _llm_extract_and_store(
        user_query: str, assistant_reply: str, user_id: str | None,
        job_id: str | None = None,
    ) -> None:
        if job_id:
            _job_set(job_id, state="running")
        try:
            extracted = await state.extractor.extract(
                user_query=user_query, assistant_reply=assistant_reply
            )
            if not extracted:
                slog.info(event="api_ingest_bg", mode="llm", count=0,
                          detail="no durable facts", user_id=user_id)
                if job_id:
                    _job_set(job_id, state="done", facts_added=0)
                return
            bundle = _aal_converter.from_extracted_items(extracted)
            items = bundle.to_memory_items()
            # Enforce storage quota on the LLM path too (Postgres only) — same
            # check sdp_ingest does. Without this, an over-quota user could grow
            # storage without bound via the background path. We log rather than
            # raise: this runs in a background task with no HTTP response left.
            if user_id and hasattr(state.memory_store, "check_quota_or_raise"):
                new_bytes = sum(len(i.content.encode("utf-8")) for i in items)
                ok, qstatus = await state.memory_store.check_quota_or_raise(
                    user_id, new_bytes,
                )
                if not ok:
                    slog.warning(
                        event="api_ingest_bg_quota_exceeded",
                        user_id=user_id,
                        used_bytes=qstatus.get("bytes_used"),
                        quota_bytes=qstatus.get("quota_bytes"),
                    )
                    return
                if qstatus.get("warning"):
                    slog.warning(
                        event="quota_soft_cap_warned",
                        user_id=user_id,
                        pct_used=qstatus.get("pct_used"),
                    )
            async with state._lock:
                await state.memory_store.add_many(items, user_id=user_id)
                await state.retriever.ingest(items)
            slog.info(event="api_ingest_bg", mode="llm", count=len(items),
                      user_id=user_id,
                      aal_well_formed=sum(1 for r in bundle if r.is_well_formed()))
            if job_id:
                _job_set(job_id, state="done", facts_added=len(items))
        except Exception as exc:  # never crash the loop; just log
            slog.warning(event="api_ingest_bg_failed",
                         error_type=type(exc).__name__, error=str(exc),
                         user_id=user_id)
            if job_id:
                _job_set(job_id, state="failed", last_error=str(exc))

    @router.post("/memory/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
        if state.extractor is None:
            raise HTTPException(
                status_code=503,
                detail="LLM ingest unavailable: MemoryExtractor not initialized "
                "(start the server with SAM/LLM enabled).",
            )
        user_id = _user_id_from(request)
        job_id = uuid.uuid4().hex
        _job_set(job_id, state="queued", facts_added=0)
        task = asyncio.create_task(
            _llm_extract_and_store(
                req.user_query, req.assistant_reply, user_id, job_id
            )
        )
        _ingest_tasks.add(task)
        task.add_done_callback(_ingest_tasks.discard)
        return IngestResponse(
            mode="llm", count=0, created=[], job_id=job_id,
            detail="extraction running in background",
        )

    @router.get("/memory/ingest/{job_id}", response_model=IngestJobResponse)
    async def ingest_job_status(job_id: str) -> IngestJobResponse:
        rec = _ingest_jobs.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown ingest job")
        return IngestJobResponse(
            job_id=rec["job_id"], state=rec["state"],
            facts_added=rec.get("facts_added", 0),
            last_error=rec.get("last_error"),
        )

    @router.post("/memory/sdp_ingest", response_model=IngestResponse)
    async def sdp_ingest(req: IngestRequest, request: Request) -> IngestResponse:
        if state.sdp is None:
            raise HTTPException(
                status_code=503, detail="sdp_ingest unavailable: SDPPipeline not initialized"
            )
        user_id = _user_id_from(request)
        # Route SDP output through AALConverter — produces AALs whose
        # sjson carries the SDP-extracted (subject, verb, object, category,
        # confidence) triples.
        bundle = _aal_converter.from_turn(
            req.user_query, req.assistant_reply, sdp_pipeline=state.sdp,
        )
        if not bundle.records:
            return IngestResponse(
                mode="sdp", count=0, created=[], detail="no pattern-detectable facts"
            )
        items = bundle.to_memory_items()
        n_high = sum(1 for r in bundle if r.importance >= 0.75)

        # Quota check (Postgres only — same pattern as /memory/ingest).
        if user_id and hasattr(state.memory_store, "check_quota_or_raise"):
            new_bytes = sum(len(i.content.encode("utf-8")) for i in items)
            ok, qstatus = await state.memory_store.check_quota_or_raise(
                user_id, new_bytes,
            )
            if not ok:
                raise HTTPException(
                    status_code=402,
                    detail={
                        "error": "storage_limit_exceeded",
                        "used_bytes": qstatus.get("bytes_used"),
                        "quota_bytes": qstatus.get("quota_bytes"),
                    },
                )
            if qstatus.get("warning"):
                slog.warning(
                    event="quota_soft_cap_warned",
                    user_id=user_id,
                    pct_used=qstatus.get("pct_used"),
                )
        async with state._lock:
            await state.memory_store.add_many(items, user_id=user_id)
            await state.retriever.ingest(items)
        slog.info(event="api_ingest", mode="sdp", count=len(items), user_id=user_id)
        return IngestResponse(
            mode="sdp",
            count=len(items),
            created=[m.id for m in items],
            detail=f"{n_high}/{len(items)} high-importance",
        )

    # ----------------------------------------------------------------------
    # Conversation memory cards (migration 019). One card per captured turn:
    # an LLM title+summary plus the atomic facts extracted from the turn (which
    # also land as normal memories so retrieval keeps improving). Heavy LLM work
    # (extraction + summarization) runs in the background like /memory/ingest;
    # the card is inserted immediately so the client gets a fast response.
    # ----------------------------------------------------------------------

    def _conv_supported() -> bool:
        return hasattr(state.memory_store, "insert_conversation")

    async def _enrich_conversation(
        conv_id: str, user_id: str | None,
        user_prompt: str, ai_response: str, model: str | None,
    ) -> None:
        try:
            extractor_client = getattr(state.extractor, "client", None) \
                if state.extractor else None
            extracted, meta = await asyncio.gather(
                state.extractor.extract(
                    user_query=user_prompt, assistant_reply=ai_response
                ) if state.extractor else _empty_extract(),
                summarize_turn(extractor_client, user_prompt, ai_response),
            )
            facts: list[dict] = []
            if extracted:
                bundle = _aal_converter.from_extracted_items(extracted)
                items = bundle.to_memory_items()
                if user_id and hasattr(state.memory_store, "check_quota_or_raise"):
                    new_bytes = sum(len(i.content.encode("utf-8")) for i in items)
                    ok, _ = await state.memory_store.check_quota_or_raise(
                        user_id, new_bytes,
                    )
                    if not ok:
                        items = []  # over quota: still keep the card, drop facts
                if items:
                    async with state._lock:
                        await state.memory_store.add_many(items, user_id=user_id)
                        await state.retriever.ingest(items)
                    facts = [{
                        "id": i.id, "content": i.content, "entity": i.entity,
                        "attribute": i.attribute, "value": i.value,
                        "confidence": i.authority_score,
                    } for i in items]
            await state.memory_store.update_conversation(
                conv_id, user_id,
                title=meta.get("title"), summary=meta.get("summary"), facts=facts,
            )
            slog.info(event="conversation_enriched", conv_id=conv_id,
                      facts=len(facts), user_id=user_id)
        except Exception as exc:  # never crash the loop
            slog.warning(event="conversation_enrich_failed", conv_id=conv_id,
                         error_type=type(exc).__name__, error=str(exc))

    async def _empty_extract() -> list:
        return []

    @router.post("/memory/conversation", response_model=ConversationCreateResponse,
                 status_code=201)
    async def create_conversation(
        req: ConversationCreateRequest, request: Request,
    ) -> ConversationCreateResponse:
        if not _conv_supported():
            raise HTTPException(
                status_code=503,
                detail="conversation cards require the Postgres backend",
            )
        if not (req.user_prompt or req.ai_response):
            raise HTTPException(
                status_code=400, detail="userPrompt or aiResponse is required",
            )
        user_id = _user_id_from(request)
        turn = f"User: {req.user_prompt}\n\nAssistant: {req.ai_response}".strip()

        # Fast: embedding + local-fallback title, insert immediately.
        try:
            vecs = await state.embedder.embed_batch([turn])
            embedding = vecs[0] if vecs else None
        except Exception:
            embedding = None
        from orchestration.ingestion.summarizer import _local_fallback
        fb = _local_fallback(req.user_prompt, req.ai_response)

        conv_id = f"conv-{uuid.uuid4().hex[:12]}"
        card = await state.memory_store.insert_conversation(
            conv_id=conv_id, user_id=user_id,
            title=fb["title"], summary=fb["summary"],
            user_prompt=req.user_prompt, ai_response=req.ai_response,
            source_url=req.source, source_model=req.model,
            facts=[], embedding=embedding,
        )

        # Background: LLM title/summary + fact extraction → memories + card patch.
        task = asyncio.create_task(
            _enrich_conversation(conv_id, user_id, req.user_prompt,
                                 req.ai_response, req.model)
        )
        _ingest_tasks.add(task)
        task.add_done_callback(_ingest_tasks.discard)

        return ConversationCreateResponse(
            id=card["id"], title=card["title"], summary=card["summary"],
            fact_count=0, job_id=conv_id,
        )

    @router.get("/memory/conversations", response_model=ConversationListResponse)
    async def list_conversations(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        q: str | None = None,
    ) -> ConversationListResponse:
        if not _conv_supported():
            return ConversationListResponse(total=0, limit=limit, offset=offset,
                                            conversations=[])
        user_id = _user_id_from(request)
        cards, total = await state.memory_store.list_conversations(
            user_id, limit=limit, offset=offset, q=q,
        )
        return ConversationListResponse(
            total=total, limit=limit, offset=offset,
            conversations=[ApiConversation(**c) for c in cards],
        )

    @router.get("/memory/conversations/{conv_id}", response_model=ApiConversation)
    async def get_conversation(conv_id: str, request: Request) -> ApiConversation:
        if not _conv_supported():
            raise HTTPException(status_code=404, detail="not found")
        user_id = _user_id_from(request)
        card = await state.memory_store.get_conversation(conv_id, user_id)
        if card is None:
            raise HTTPException(status_code=404, detail=f"conversation {conv_id!r} not found")
        return ApiConversation(**card)

    @router.delete("/memory/conversations/{conv_id}")
    async def delete_conversation(conv_id: str, request: Request) -> dict:
        if not _conv_supported():
            raise HTTPException(status_code=404, detail="not found")
        user_id = _user_id_from(request)
        ok = await state.memory_store.delete_conversation(conv_id, user_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"conversation {conv_id!r} not found")
        return {"deleted": True, "id": conv_id}

    # ----------------------------------------------------------------------

    # Quota — Phase 6 — only fires on the Postgres backend.
    # ----------------------------------------------------------------------

    class QuotaStatusResponse(BaseModel):
        user_id: str
        plan: str | None = None
        quota_bytes: int
        bytes_used: int
        pct_used: float
        memory_count: int = 0
        warned_at_90pct: bool = False

    @router.get("/me/quota", response_model=QuotaStatusResponse)
    async def my_quota(request: Request) -> QuotaStatusResponse:
        """Return the authenticated caller's storage quota status. Used by
        the UI to render the 'X% of 1 GB used' indicator."""
        user_id = _user_id_from(request)
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="quota only applies to authenticated user requests"
                       " (not admin)",
            )
        if not hasattr(state.memory_store, "get_quota_status"):
            # JSONL backend has no per-user accounting — return a fake
            # 'unlimited' status so the UI still works.
            return QuotaStatusResponse(
                user_id=user_id, plan="dev",
                quota_bytes=0, bytes_used=0, pct_used=0.0,
            )
        status = await state.memory_store.get_quota_status(user_id)
        return QuotaStatusResponse(
            user_id=status["user_id"],
            plan=status.get("plan"),
            quota_bytes=int(status.get("quota_bytes") or 0),
            bytes_used=int(status.get("bytes_used") or 0),
            pct_used=float(status.get("pct_used") or 0.0),
            memory_count=int(status.get("memory_count") or 0),
            warned_at_90pct=bool(status.get("warned_at_90pct")),
        )

    # ----------------------------------------------------------------------
    # Personalized MCP install surface — issues per-user API keys and returns
    # **opaque** install links (the raw key is never in a human-readable
    # field). Cursor gets a deep link with the key base64-embedded; the rest
    # get download URLs that stream a pre-baked config file. The extension
    # gets a chrome.identity-compatible SSO flow.
    # ----------------------------------------------------------------------
    _key_store_cache: dict = {"store": None}

    async def _key_store():
        if _key_store_cache["store"] is None:
            from orchestration.storage import make_user_key_store
            _key_store_cache["store"] = await make_user_key_store()
        return _key_store_cache["store"]

    def _mcp_url() -> str:
        return os.environ.get("GML_PUBLIC_MCP_URL", "https://akhrots.com/mcp")

    def _public_base() -> str:
        # Default to akhrots.com; override with GML_PUBLIC_BASE_URL if you front
        # the API at a different origin.
        return os.environ.get("GML_PUBLIC_BASE_URL", "https://akhrots.com").rstrip("/")

    def _server_block(api_key: str) -> dict:
        return {
            "url": _mcp_url(),
            "headers": {"Authorization": f"Bearer {api_key}"},
        }

    def _cursor_deeplink(api_key: str) -> str:
        cfg_b64 = base64.urlsafe_b64encode(
            json.dumps(_server_block(api_key)).encode("utf-8")
        ).decode("ascii")
        return (
            "cursor://anysphere.cursor-deeplink/mcp/install"
            f"?name=akhrot-memory&config={cfg_b64}"
        )

    def _vscode_deeplink(api_key: str) -> str:
        # VS Code's MCP-install handler (Copilot Chat MCP). The shape mirrors
        # Cursor: a single server entry base64-encoded into the URL.
        entry = {
            "name": "akhrot-memory",
            "type": "http",
            "url": _mcp_url(),
            "headers": {"Authorization": f"Bearer {api_key}"},
        }
        cfg_b64 = base64.urlsafe_b64encode(
            json.dumps(entry).encode("utf-8")
        ).decode("ascii")
        return f"vscode://GitHub.copilot-chat/mcp/install?config={cfg_b64}"

    def _config_file_bytes(api_key: str) -> bytes:
        """The {mcpServers:{akhrot-memory:{...}}} config most IDEs accept."""
        cfg = {"mcpServers": {"akhrot-memory": _server_block(api_key)}}
        return json.dumps(cfg, indent=2).encode("utf-8")

    # ---------- Claude Desktop / Windsurf .dxt baking ----------------------
    # The .dxt bundle ships at /var/www/akhrots/static/akhrot-memory.dxt as a
    # static template. Its manifest.json declares ${user_config.token} which
    # makes Claude Desktop prompt for the bearer on install. We replace that
    # placeholder with a freshly-issued per-user key and strip the user_config
    # block entirely, so the IDE installs the extension silently AND it's
    # already authenticated to akhrots.com/mcp as this user. No paste step.

    def _dxt_template_path() -> Path:
        return Path(os.environ.get(
            "AKHROT_CLAUDE_DXT_TEMPLATE",
            "/var/www/akhrots/static/akhrot-memory.dxt",
        ))

    def _bake_dxt_bytes(api_key: str) -> bytes:
        """Open the template .dxt, inject the bearer into the manifest's env,
        drop the user_config block, return the re-zipped bytes."""
        src = _dxt_template_path()
        if not src.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Claude .dxt template missing at {src}",
            )
        out = io.BytesIO()
        with zipfile.ZipFile(src, "r") as zin, \
             zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in zin.namelist():
                data = zin.read(name)
                if name == "manifest.json":
                    m = json.loads(data.decode("utf-8"))
                    env = m.setdefault("server", {}).setdefault("mcp_config", {}).setdefault("env", {})
                    env["GML_TOKEN"] = api_key
                    env["GML_MCP_URL"] = _mcp_url()
                    # Removing user_config tells Claude Desktop "no UI prompt
                    # needed; all required values are already resolved".
                    m.pop("user_config", None)
                    data = json.dumps(m, indent=2).encode("utf-8")
                zout.writestr(name, data)
        return out.getvalue()

    def _dxt_response(api_key: str) -> Response:
        return Response(
            content=_bake_dxt_bytes(api_key),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="akhrot-memory.mcpb"',
                "Cache-Control": "no-store",
            },
        )

    # ---------- Nuts (macOS menu-bar companion) ----------------------
    # Replaces the previous Chrome-extension install vector. We serve a zip
    # holding the prebuilt .dmg plus a per-user akhort-config.json. Nuts's
    # first-launch code reads the json, stores the bearer in macOS Keychain,
    # then deletes the source file - the user never sees a sign-in screen.

    def _nuts_dmg_path() -> Path:
        return Path(os.environ.get(
            "AKHROT_NUTS_DMG",
            "/var/www/akhrots/static/nuts.dmg",
        ))

    def _bake_nuts_zip(api_key: str) -> bytes:
        src = _nuts_dmg_path()
        if not src.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Nuts build missing at {src}; upload the prebuilt .dmg.",
            )
        config = json.dumps(
            {"token": api_key, "url": _mcp_url()},
            indent=2,
        ).encode("utf-8")
        readme = (
            "Akhort + Nuts - quick install\n\n"
            "1. Open the .dmg and drag Nuts into Applications.\n"
            "2. Launch Nuts.\n"
            "3. Done - you're already signed in.\n\n"
            "akhort-config.json carries your sign-in. Nuts reads it on\n"
            "first launch and removes it. You don't need to open it.\n"
        ).encode("utf-8")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("Nuts.dmg", src.read_bytes())
            z.writestr("akhort-config.json", config)
            z.writestr("README.txt", readme)
        return buf.getvalue()

    # ---------- Nuts for Windows -----------------------------------------
    # Same shape as the macOS bake above, only the binary differs: we ship a
    # PyInstaller-built Nuts.exe (or a small installer .zip from the
    # nuts-windows repo) plus the per-user config json. The Windows
    # bootstrap reads %USERPROFILE%\Downloads\akhort-config.json on first
    # launch (see nuts_windows.bootstrap).

    def _nuts_windows_path() -> Path:
        return Path(os.environ.get(
            "AKHROT_NUTS_WINDOWS_EXE",
            "/var/www/akhrots/static/Nuts.exe",
        ))

    def _bake_nuts_windows_zip(api_key: str) -> bytes:
        src = _nuts_windows_path()
        if not src.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Nuts (Windows) build missing at {src}; upload the prebuilt binary.",
            )
        config = json.dumps(
            {"token": api_key, "url": _mcp_url()},
            indent=2,
        ).encode("utf-8")
        readme = (
            "Akhort + Nuts for Windows - quick install\n\n"
            "1. Extract this zip anywhere (e.g. Desktop).\n"
            "2. Double-click Nuts.exe to launch.\n"
            "3. SmartScreen may warn on first run - click More info -> Run anyway.\n"
            "4. A tray icon appears. Hold Ctrl+Alt to talk.\n\n"
            "akhort-config.json carries your sign-in. Nuts reads it on first\n"
            "launch and removes it. You don't need to open it.\n"
        ).encode("utf-8")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            # If the static artifact is a .zip itself (e.g. an installer
            # bundle), copy its members. Otherwise treat it as the .exe.
            if src.suffix.lower() == ".zip":
                with zipfile.ZipFile(src, "r") as zin:
                    for name in zin.namelist():
                        z.writestr(name, zin.read(name))
            else:
                z.writestr("Nuts.exe", src.read_bytes())
            z.writestr("akhort-config.json", config)
            z.writestr("README.txt", readme)
        return buf.getvalue()

    class McpConfigResponse(BaseModel):
        """Install bundle for the signed-in user.

        Note: this response never carries the raw API key as a readable field.
        The key is embedded (base64) inside `cursor_deeplink` / `vscode_deeplink`
        and is what the download URLs serve when fetched. UI is expected to
        render buttons against these opaque URLs, not display them as text.
        """

        client: str
        server_url: str
        cursor_deeplink: str
        vscode_deeplink: str
        # Each `install_url_*` streams a downloadable JSON config file.
        install_url_claude_desktop: str
        install_url_windsurf: str
        install_url_generic: str
        # Per-user zip: Nuts.dmg + akhort-config.json. macOS only.
        install_url_nuts: str
        # Per-user zip: Nuts.exe + akhort-config.json. Windows only.
        install_url_nuts_windows: str
        # OpenAI Codex — one-click installers (drop the stdio bridge + merge
        # ~/.codex/config.toml) and a /plugins bundle. Each bakes a per-user
        # token, like the rest of this surface.
        install_url_codex_windows: str
        install_url_codex_macos: str
        install_url_codex_plugin: str
        # Extension download URL - kept for any straggler consumers; the
        # dashboard no longer renders an extension block (Nuts replaces it).
        extension_download_url: str

    @router.post("/me/mcp-config", response_model=McpConfigResponse)
    async def my_mcp_config(
        request: Request, client: str = Query("generic"),
    ) -> McpConfigResponse:
        user_id = _user_id_from(request)
        if not user_id:
            raise HTTPException(status_code=401, detail="sign in to get an MCP config")
        store = await _key_store()
        if not hasattr(store, "issue"):
            raise HTTPException(
                status_code=501,
                detail="MCP config requires the Postgres backend (per-user keys)",
            )
        rec = await store.issue(user_id, label=f"mcp-{client}")
        slog.info(event="mcp_config_issued", user_id=user_id, client=client)
        base = _public_base()
        return McpConfigResponse(
            client=client,
            server_url=_mcp_url(),
            cursor_deeplink=_cursor_deeplink(rec.key),
            vscode_deeplink=_vscode_deeplink(rec.key),
            # Cookie-auth on these — the signed-in user's browser fetches and
            # the response includes the file as an attachment. A fresh per-call
            # key is minted on each fetch so URLs aren't long-lived secrets.
            install_url_claude_desktop=f"{base}/api/me/install/claude-desktop",
            install_url_windsurf=f"{base}/api/me/install/windsurf",
            install_url_generic=f"{base}/api/me/install/generic",
            install_url_nuts=f"{base}/api/me/install/nuts",
            install_url_nuts_windows=f"{base}/api/me/install/nuts-windows",
            install_url_codex_windows=f"{base}/api/me/install/codex-windows",
            install_url_codex_macos=f"{base}/api/me/install/codex-macos",
            install_url_codex_plugin=f"{base}/api/me/install/codex-plugin",
            extension_download_url=f"{base}/api/me/extension-download",
        )

    # ── Per-IDE install file downloads ─────────────────────────────────────
    # These stream a single config file. Each call mints a fresh per-user key
    # — there's no point pretending the URL is idempotent, the file *is* the
    # secret. The dashboard fires these via <a download> so the browser saves
    # them to the user's downloads folder, where they drag/drop them into the
    # IDE's MCP config path (shown in the dialog).
    def _attachment(name: str, body: bytes) -> Response:
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{name}"',
                "Cache-Control": "no-store",
            },
        )

    async def _issue_or_401(request: Request, label: str):
        user_id = _user_id_from(request)
        if not user_id:
            raise HTTPException(status_code=401, detail="sign in to install")
        store = await _key_store()
        if not hasattr(store, "issue"):
            raise HTTPException(status_code=501,
                                detail="install requires the Postgres backend")
        return user_id, await store.issue(user_id, label=label)

    @router.get("/me/install/claude-desktop")
    async def install_claude_desktop(request: Request) -> Response:
        # Returns the per-user .dxt bundle. Claude Desktop double-clicks the
        # downloaded file and installs the MCP without prompting for a token
        # (we baked the bearer into the manifest's env on the fly).
        _, rec = await _issue_or_401(request, "install-claude-desktop")
        return _dxt_response(rec.key)

    @router.get("/me/install/windsurf")
    async def install_windsurf(request: Request) -> Response:
        # Same .dxt; Windsurf accepts the format via its DXT/MCPB support.
        # If your Windsurf build pre-dates DXT support, this still works:
        # the user can drag the .dxt's manifest into mcp_config.json manually.
        _, rec = await _issue_or_401(request, "install-windsurf")
        return _dxt_response(rec.key)

    @router.get("/me/install/nuts")
    async def install_nuts(request: Request) -> Response:
        # Per-user zip download. Nuts picks up the bundled
        # akhort-config.json on first launch and signs in silently. Returns
        # 503 if no prebuilt .dmg has been uploaded yet (see _nuts_dmg_path).
        _, rec = await _issue_or_401(request, "install-nuts")
        return Response(
            content=_bake_nuts_zip(rec.key),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="Akhort-Nuts.zip"',
                "Cache-Control": "private, no-store",
            },
        )

    @router.get("/me/install/nuts-windows")
    async def install_nuts_windows(request: Request) -> Response:
        # Windows counterpart. Same per-user zip shape (Nuts.exe +
        # akhort-config.json). 503 until you upload the prebuilt exe -
        # see _nuts_windows_path() for the expected disk path / env var.
        _, rec = await _issue_or_401(request, "install-nuts-windows")
        return Response(
            content=_bake_nuts_windows_zip(rec.key),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="Akhort-Nuts-Windows.zip"',
                "Cache-Control": "private, no-store",
            },
        )

    @router.get("/me/install/generic")
    async def install_generic(request: Request) -> Response:
        _, rec = await _issue_or_401(request, "install-generic")
        return _attachment("akhrot-mcp-config.json", _config_file_bytes(rec.key))

    # ── OpenAI Codex installers ───────────────────────────────────────────
    # Codex speaks stdio MCP, so it can't consume the streamable-HTTP /mcp
    # endpoint directly the way Cursor/VS Code do. These bake the per-user
    # token into the vendored stdio↔HTTP bridge: the one-click installers drop
    # it into ~/.codex/akhrot-memory/index.js and merge an idempotent
    # [mcp_servers.akhrot-memory] block into ~/.codex/config.toml; the plugin
    # zip is the same bridge wired through Codex's /plugins. Self-contained —
    # unlike nuts/claude these need no pre-uploaded artifact (the bridge ships
    # in the repo), so they never 503.
    def _codex_artifact(builder, key: str, name: str, media_type: str) -> Response:
        try:
            body = builder(key, _mcp_url())
        except FileNotFoundError as exc:
            # The bridge ships in the repo; a miss means a packaging/deploy bug.
            # Surface it like the other missing-artifact installers (503) rather
            # than a raw 500, so the dashboard can message it.
            raise HTTPException(
                status_code=503, detail=f"Codex bridge missing from deploy: {exc}"
            )
        return Response(
            content=body,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{name}"',
                "Cache-Control": "no-store",
            },
        )

    @router.get("/me/install/codex-windows")
    async def install_codex_windows(request: Request) -> Response:
        _, rec = await _issue_or_401(request, "install-codex-windows")
        return _codex_artifact(
            codex_connector.windows_installer, rec.key,
            "install-akhrot-codex.cmd", "application/octet-stream",
        )

    @router.get("/me/install/codex-macos")
    async def install_codex_macos(request: Request) -> Response:
        # .command double-clicks in Finder; it's plain bash, so Linux works too.
        _, rec = await _issue_or_401(request, "install-codex-macos")
        return _codex_artifact(
            codex_connector.unix_installer, rec.key,
            "install-akhrot-codex.command", "application/octet-stream",
        )

    @router.get("/me/install/codex-plugin")
    async def install_codex_plugin(request: Request) -> Response:
        _, rec = await _issue_or_401(request, "install-codex-plugin")
        return _codex_artifact(
            codex_connector.plugin_zip, rec.key,
            "akhrot-memory-codex-plugin.zip", "application/zip",
        )

    # ── Chrome extension binding ──────────────────────────────────────────
    # The dashboard (which has the user's JWT in localStorage) calls this
    # with a Bearer token to mint a fresh long-lived MCP key for the
    # extension, then pushes the key to the installed extension via
    # `chrome.runtime.sendMessage(EXTENSION_ID, ...)` — the extension's
    # `externally_connectable` allowlists akhrots.com so it accepts that
    # message. The user never sees the key text; it lives only in extension
    # storage + this server.
    class ExtensionBindResponse(BaseModel):
        user_id: str
        token: str

    @router.post("/me/extension-bind", response_model=ExtensionBindResponse)
    async def extension_bind(request: Request) -> ExtensionBindResponse:
        user_id = _user_id_from(request)
        if not user_id:
            raise HTTPException(status_code=401, detail="sign in to bind extension")
        store = await _key_store()
        if not hasattr(store, "issue"):
            raise HTTPException(status_code=501,
                                detail="extension binding requires the Postgres backend")
        rec = await store.issue(user_id, label="extension")
        slog.info(event="extension_bind", user_id=user_id)
        return ExtensionBindResponse(user_id=user_id, token=rec.key)

    # ── Chrome extension download (one zip; auth required) ───────────────
    # The dashboard wires "Download extension" to this URL. We require a
    # signed-in user so anonymous scrapers don't pull the artifact.
    @router.get("/me/extension-download")
    async def extension_download(request: Request) -> Response:
        user_id = _user_id_from(request)
        if not user_id:
            raise HTTPException(status_code=401, detail="sign in to download")
        path = Path(
            os.environ.get("AKHROT_EXTENSION_ZIP",
                           "/var/www/akhrots/static/akhrot-extension.zip")
        )
        if not path.exists():
            raise HTTPException(status_code=503, detail="extension build not available")
        slog.info(event="extension_download", user_id=user_id, size=path.stat().st_size)
        return Response(
            content=path.read_bytes(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="akhrot-extension.zip"',
                "Cache-Control": "no-store",
            },
        )

    return router
