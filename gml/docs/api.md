# GML Web API (`/api`)

Stable HTTP contract consumed by the Next.js web UI (`web/`). Mounted by
`gml serve` (FastAPI, default port `8000`). Every route reuses existing
orchestration logic; the only derived data is the cluster/graph view
(`orchestration/graph_projection.py`, KMeans + kNN over memory vectors).

Start the backend:

```bash
gml serve --port 8000            # full pipeline (needs Ollama for SAM/LLM ingest)
gml serve --port 8000 --no-sam-llm --stub-client   # LLM-free dev mode
```

Interactive docs: `http://localhost:8000/docs` (OpenAPI).

## Schema mapping (backend → API)

The persisted `MemoryItem` has no `confidence`/`importance`/`cluster_id`.
The API derives them:

| API field    | Source |
|--------------|--------|
| `importance` | `MemoryItem.authority_score` |
| `confidence` | `raw_metadata["confidence"]` (SDP) → falls back to `authority_score` |
| `cluster_id` | KMeans label (≤6 clusters), cached, recomputed on add/remove |

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET | `/api/health` | liveness + memory count + embedder version |
| GET | `/api/memories?cluster=&limit=&offset=` | paginated list; `total` for pagination |
| GET | `/api/memories/{id}` | memory + `relationships` (similarity kNN + shared-entity) |
| GET | `/api/memories/graph?depth=` | `{nodes, edges}` for the 3D graph. `depth` is accepted but the graph is global (kNN edges); reserved for future focal-node expansion |
| DELETE | `/api/memories/{id}` | forget — rewrites store + drops vector. 404 if absent |
| GET | `/api/clusters` | `{id, label, centroid{x,y}, size, color_hint}` per cluster |
| POST | `/api/memory/recall` | `{query, top_k}` → raw vector hits + score + heuristic `why` |
| POST | `/api/memory/ingest` | `{user_query, assistant_reply}` → LLM extract. **503** if no extractor (Ollama off) |
| POST | `/api/memory/sdp_ingest` | `{user_query, assistant_reply}` → fast regex SDP (always available) |
| POST | `/api/memory/trace` | `{text}` → per-stage pipeline trace (same payload as `/viz/run`) |
| POST | `/api/memory/recall/stream` | SSE: `event: stage` per completed pipeline stage (real timing), then `event: done` with reranked `results` + SAM reasoning. Drives the UI's 7-stage recall indicator. |
| POST | `/api/memory/trace/stream` | SSE: same stages streamed, `event: done` carries the full trace payload. |
| GET | `/api/memory/synthesize?query=` | assembled context string (`context` field) |

SSE frames are `event: <stage\|done\|error>\n` + `data: <json>\n\n`. Both stream
endpoints reuse `stream_pipeline_trace()` (an async generator in `server.py`);
the non-streaming `/trace` and `/viz/run` collect the same generator, so output
stays identical across all four.

## Known limits / not-yet-production

- **No auth / rate limiting / TLS.** CORS is `*` (dev). Put a gateway in front for production.
- **`recall.why`** is a token-overlap heuristic, not SAM reasoning. Real reasoning
  is only produced on the full pipeline path (`/synthesize`, `/trace`).
- **Reranker latency**: the FT cross-encoder reranker stage can take several
  seconds — a pre-existing pipeline characteristic, surfaced in `/trace` timings.
- Clusters/graph are computed in-process and cached by memory-set signature;
  fine for thousands of memories, not for very large corpora.
