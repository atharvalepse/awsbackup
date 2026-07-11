"""Thin MCP-over-stdio proxy onto the GML HTTP API.

The full MCP server (:mod:`orchestration.mcp_server`) boots the entire
pipeline in-process — SentenceTransformer embedder, cross-encoder rerankers,
SAM, extractor — which costs 1.5–2 GB RSS *per process*. Behind the relay
connector every host session gets its own ``gml mcp`` child, so a handful of
Claude Desktop sessions can exhaust the box (observed: 7 children ≈ 9 GB,
swap full, OOM kills).

This module is the fix: when ``GML_MCP_PROXY_URL`` is set (stdio transport
only), ``gml mcp`` serves the same tool surface but forwards every call to
the already-running HTTP API (``gml serve``) on that URL. The models load
once, in the API server; each MCP child stays ~30 MB.

Auth: ``GML_API_KEY`` is sent as ``X-API-Key``. When it is the master key and
``GML_MCP_USER`` is set (the relay connector sets it per host session), the
proxy adds ``X-GML-Act-As`` so the API scopes reads/writes to that tenant —
the same scoping the in-process server gets via the GML_MCP_USER fallback.

Imports here must stay light: no torch, no orchestration.pipeline. That is
the entire point of the module.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


def _base_url() -> str:
    return os.environ.get("GML_MCP_PROXY_URL", "http://127.0.0.1:8000").rstrip("/")


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    key = os.environ.get("GML_API_KEY", "").strip()
    if key:
        headers["X-API-Key"] = key
    user = os.environ.get("GML_MCP_USER", "").strip()
    if user:
        # Honored only for the master key (the API 403s otherwise — loud
        # beats writing into the wrong tenant).
        headers["X-GML-Act-As"] = user
    return headers


_client_cache: dict[str, httpx.AsyncClient] = {}


def _client() -> httpx.AsyncClient:
    # One client per event loop run; synthesize can sit behind a CPU LLM for
    # a while, so the read timeout is generous.
    cli = _client_cache.get("instance")
    if cli is None or cli.is_closed:
        cli = httpx.AsyncClient(
            base_url=_base_url(),
            headers=_headers(),
            timeout=httpx.Timeout(180.0, connect=5.0),
        )
        _client_cache["instance"] = cli
    return cli


def _unreachable(exc: Exception) -> str:
    return (
        f"GML API at {_base_url()} unreachable "
        f"({type(exc).__name__}: {exc}). Is the HTTP server (`gml serve` / "
        "gml-api.service) running?"
    )


def _error_body(resp: httpx.Response) -> str:
    try:
        return json.dumps(resp.json().get("detail", resp.json()))
    except Exception:
        return resp.text[:300]


def _format_memories(records: list[dict[str, Any]]) -> str:
    """Same human-readable shape the in-process server returns."""
    if not records:
        return "(no relevant memories found)"
    lines = []
    for i, r in enumerate(records, start=1):
        head = f"{i}. [{r.get('source', '?')}]"
        if r.get("entity"):
            head += f" {r['entity']}"
            if r.get("attribute"):
                head += f"/{r['attribute']}"
            if r.get("value"):
                head += f" = {r['value']}"
        lines.append(head)
        lines.append(f"   {r.get('content', '')}")
        if r.get("similarity") is not None:
            lines.append(f"   (relevance: {r['similarity']:.2f})")
        lines.append(f"   id: {r.get('id')}")
    return "\n".join(lines)


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
)


@mcp.tool()
async def query(text: str) -> str:
    """Run the FULL GML pipeline for one user turn.

    Stages: Classifier → Embedder → Retriever → Reranker → SAM →
    Assembler → Translator (executed by the GML API server). Returns the
    formatted-context block the calling AI should paraphrase into its
    answer to the user.

    Args:
        text: The user's question, verbatim.
    """
    try:
        resp = await _client().get(
            "/api/memory/synthesize", params={"query": text}
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"query failed ({resp.status_code}): {_error_body(resp)}"
    body = resp.json()
    return (
        f"<gml_context>\n{body['context']}\n</gml_context>\n"
        f"[items_included={body.get('items_included')} via=gml-http-api]"
    )


@mcp.tool()
async def ingest(user_query: str, assistant_reply: str) -> str:
    """Persist durable facts from a completed user→assistant exchange.

    Runs the LLM MemoryExtractor on the turn (in the GML API server) and
    persists extracted facts so the next `query` call sees them. Call this
    AFTER your reply, with both sides of the exchange. If nothing durable
    is found, nothing is saved — by design, not an error.

    Args:
        user_query: What the user asked, verbatim.
        assistant_reply: What you just answered.
    """
    try:
        resp = await _client().post(
            "/api/memory/ingest",
            json={"user_query": user_query, "assistant_reply": assistant_reply},
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"ingest failed ({resp.status_code}): {_error_body(resp)}"
    job_id = resp.json().get("job_id")
    if not job_id:
        return f"ingest accepted: {resp.json().get('detail') or 'submitted'}"

    # Extraction runs in the API server's background (~7s on the local
    # LLM). Poll briefly so normal calls return a real outcome; fall back
    # to the job id if it is still running.
    for _ in range(25):
        await asyncio.sleep(1.0)
        try:
            jr = await _client().get(f"/api/memory/ingest/{job_id}")
        except httpx.HTTPError as exc:
            return _unreachable(exc)
        if jr.status_code != 200:
            break
        job = jr.json()
        if job["state"] == "done":
            n = job.get("facts_added", 0)
            if n == 0:
                return "ingest: no durable facts extracted"
            return f"ingest: saved {n} memories"
        if job["state"] == "failed":
            return f"ingest failed: {job.get('last_error')}"
    return f"ingest: extraction still running in background (job {job_id})"


@mcp.tool()
async def sdp_ingest(user_query: str, assistant_reply: str) -> str:
    """Persist facts using the LIGHTWEIGHT regex-based SDP pipeline.

    Same idea as `ingest()` but no LLM: ~100x faster, catches only
    pattern-detectable facts (tech stack, versions, ports, URLs, people,
    supersession verbs). Use it for the fast path on clearly-factual
    turns; use `ingest()` when nuance matters.

    Args:
        user_query: What the user said, verbatim.
        assistant_reply: What you just answered.
    """
    try:
        resp = await _client().post(
            "/api/memory/sdp_ingest",
            json={"user_query": user_query, "assistant_reply": assistant_reply},
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"sdp_ingest failed ({resp.status_code}): {_error_body(resp)}"
    body = resp.json()
    if body["count"] == 0:
        return f"sdp_ingest: {body.get('detail') or 'no pattern-detectable facts'}"
    ids = ", ".join(body["created"][:5])
    more = "" if body["count"] <= 5 else f" (+{body['count'] - 5} more)"
    return (
        f"sdp_ingest: saved {body['count']} memories — {ids}{more}"
        + (f"\n  {body['detail']}" if body.get("detail") else "")
    )


@mcp.tool()
async def recall(query: str, top_k: int = 5) -> str:
    """Low-level retrieval bypass. Skip the pipeline; just search the index.

    Prefer `query(text)` for normal use. `recall` returns raw reranked
    hits without SAM reasoning or target-aware formatting.

    Args:
        query: The user's question.
        top_k: How many memories to return. Default 5.
    """
    try:
        resp = await _client().post(
            "/api/memory/recall",
            json={"query": query, "top_k": max(1, min(50, top_k))},
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"recall failed ({resp.status_code}): {_error_body(resp)}"
    records = []
    for r in resp.json()["results"]:
        m = dict(r["memory"])
        m["similarity"] = r.get("score")
        records.append(m)
    return _format_memories(records)


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

    Prefer `ingest(user_query, assistant_reply)` — the extractor usually
    finds multiple structured facts per turn. Use `remember` only for a
    single pre-formed claim.

    Args:
        content: One full-sentence claim worth remembering, third person.
        entity: Subject of the claim.
        attribute: Property of the entity.
        value: Value of the attribute.
        source: Where this came from. Default "conversation".
        authority_score: 0-1 trust score. Default 0.7.
    """
    try:
        resp = await _client().post(
            "/api/memories",
            json={
                "content": content,
                "entity": entity,
                "attribute": attribute,
                "value": value,
                "source": source,
                "authority_score": max(0.0, min(1.0, authority_score)),
            },
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code not in (200, 201):
        return f"remember failed ({resp.status_code}): {_error_body(resp)}"
    body = resp.json()
    return f"Saved memory {body['id']}: {content!r}"


@mcp.tool()
async def forget(memory_id: str) -> str:
    """Remove a memory from the store.

    Args:
        memory_id: The id returned by `recall` or `list_memories`.
    """
    try:
        resp = await _client().delete(f"/api/memories/{memory_id}")
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code == 404:
        return f"No memory found with id {memory_id!r}"
    if resp.status_code != 200:
        return f"forget failed ({resp.status_code}): {_error_body(resp)}"
    return f"Forgot memory {memory_id!r}"


@mcp.tool()
async def list_memories(entity: str | None = None, limit: int = 20) -> str:
    """Browse memories in the store.

    Args:
        entity: If set, only return memories where ``entity`` matches.
        limit: Max records to return. Default 20.
    """
    try:
        resp = await _client().get("/api/memories", params={"limit": 500})
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"list_memories failed ({resp.status_code}): {_error_body(resp)}"
    records = resp.json()["memories"]
    if entity:
        records = [r for r in records if r.get("entity") == entity]
    records = records[-max(1, limit):]
    return _format_memories(records) if records else "(no memories yet)"


@mcp.tool()
async def improve_query(text: str) -> str:
    """Return the query unchanged.

    Query improvement happens inside the API server's pipeline run; this
    standalone heuristic is not exposed over HTTP, so proxy mode is
    passthrough (same behavior as the heuristic-only local server).
    """
    return text


async def _status_text() -> str:
    try:
        resp = await _client().get("/api/health")
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"status failed ({resp.status_code}): {_error_body(resp)}"
    body = resp.json()
    user = os.environ.get("GML_MCP_USER", "").strip() or "(unscoped)"
    return (
        f"GML memory server (HTTP proxy mode)\n"
        f"  api:       {_base_url()}\n"
        f"  status:    {body.get('status')}\n"
        f"  version:   {body.get('version')}\n"
        f"  embedder:  {body.get('embedder')}\n"
        f"  memories:  {body.get('memories')}\n"
        f"  acting as: {user}\n"
    )


@mcp.tool()
async def status() -> str:
    """Report GML server state — proxy target, version, memory count."""
    return await _status_text()


@mcp.tool()
async def trace(text: str) -> str:
    """Run the full pipeline and return a stage-by-stage breakdown.

    Executed by the GML API server; returns its structured trace as JSON.

    Args:
        text: The user's question, verbatim.
    """
    try:
        resp = await _client().post("/api/memory/trace", json={"text": text})
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    if resp.status_code != 200:
        return f"trace failed ({resp.status_code}): {_error_body(resp)}"
    out = json.dumps(resp.json(), indent=2, ensure_ascii=False)
    if len(out) > 12000:
        out = out[:12000] + "\n... (truncated)"
    return out


@mcp.tool()
async def analyze(text: str, history: str | None = None) -> str:
    """Not available in proxy mode.

    `analyze` needs a direct LLM client. Run the full local server
    (unset GML_MCP_PROXY_URL) or use `trace(text)` for pipeline insight.
    """
    return (
        "analyze unavailable in HTTP proxy mode — use `trace(text)` for "
        "pipeline insight, or run the full local server (unset "
        "GML_MCP_PROXY_URL)."
    )


@mcp.tool()
async def diag() -> str:
    """Storage diagnostic (proxy mode: API health summary)."""
    return await _status_text()


def run() -> None:
    """Entry point: serve the proxy tool surface over stdio."""
    sys.stderr.write(
        f"gml-memory MCP proxy ready (stdio → {_base_url()}, "
        f"user={os.environ.get('GML_MCP_USER', '') or 'unscoped'})\n"
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
