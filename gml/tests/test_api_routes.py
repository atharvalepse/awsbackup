"""Tests for the /api web-UI surface (build_api_router).

Run LLM-free: StubEmbedder + stub client + SAM(reasoner=None). The SDP path
needs no LLM, so sdp_ingest is exercised end-to-end; the LLM ingest path is
asserted to degrade cleanly (503) when no extractor is configured.
"""
import pytest
from fastapi.testclient import TestClient

from orchestration.embedder import StubEmbedder
from orchestration.server import build_app, build_default_state


async def _client(tmp_path) -> TestClient:
    state = await build_default_state(
        embedder=StubEmbedder(dim=384),
        memory_path=tmp_path / "memories.jsonl",
        enable_sam_llm=False,
        stub_client=True,
    )
    return TestClient(build_app(state))


@pytest.mark.asyncio
async def test_api_health(tmp_path):
    client = await _client(tmp_path)
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["memories"] > 0  # seeded fixture


@pytest.mark.asyncio
async def test_api_memories_list_and_pagination(tmp_path):
    client = await _client(tmp_path)
    r = client.get("/api/memories?limit=2&offset=0")
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"total", "limit", "offset", "memories"} <= body.keys()
    assert body["limit"] == 2
    assert len(body["memories"]) <= 2
    m = body["memories"][0]
    # schema mapping is present
    assert {"id", "confidence", "importance", "cluster_id", "timestamp"} <= m.keys()
    assert 0.0 <= m["confidence"] <= 1.0
    assert 0.0 <= m["importance"] <= 1.0


@pytest.mark.asyncio
async def test_api_clusters_and_graph(tmp_path):
    client = await _client(tmp_path)
    rc = client.get("/api/clusters")
    assert rc.status_code == 200, rc.text
    clusters = rc.json()["clusters"]
    assert clusters, "expected at least one cluster"
    c0 = clusters[0]
    assert {"id", "label", "centroid", "size", "color_hint"} <= c0.keys()
    assert c0["color_hint"].startswith("cluster-")

    rg = client.get("/api/memories/graph?depth=2")
    assert rg.status_code == 200, rg.text
    graph = rg.json()
    assert graph["nodes"], "expected graph nodes"
    n0 = graph["nodes"][0]
    assert {"id", "cluster_id", "importance", "val", "x", "y"} <= n0.keys()
    # edges reference real node ids
    node_ids = {n["id"] for n in graph["nodes"]}
    for e in graph["edges"]:
        assert e["source"] in node_ids and e["target"] in node_ids
        assert -1.0 <= e["weight"] <= 1.0

    # cluster filter returns a subset
    cid = c0["id"]
    rf = client.get(f"/api/memories?cluster={cid}")
    assert rf.status_code == 200
    assert all(m["cluster_id"] == cid for m in rf.json()["memories"])


@pytest.mark.asyncio
async def test_api_memory_detail_and_404(tmp_path):
    client = await _client(tmp_path)
    listing = client.get("/api/memories?limit=1").json()["memories"]
    mid = listing[0]["id"]
    r = client.get(f"/api/memories/{mid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["memory"]["id"] == mid
    assert isinstance(body["relationships"], list)

    r404 = client.get("/api/memories/does-not-exist")
    assert r404.status_code == 404


@pytest.mark.asyncio
async def test_api_recall(tmp_path):
    client = await _client(tmp_path)
    r = client.post("/api/memory/recall", json={"query": "what port", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "what port"
    assert isinstance(body["results"], list)
    for res in body["results"]:
        assert {"memory", "score"} <= res.keys()
        assert -1.0 <= res["score"] <= 1.0


@pytest.mark.asyncio
async def test_api_synthesize_and_trace(tmp_path):
    client = await _client(tmp_path)
    rs = client.get("/api/memory/synthesize", params={"query": "what do you know"})
    assert rs.status_code == 200, rs.text
    assert "context" in rs.json()

    rt = client.post("/api/memory/trace", json={"text": "what do you know"})
    assert rt.status_code == 200, rt.text
    trace = rt.json()
    assert "stages" in trace and trace["stages"]
    assert {"query", "annotations", "formatted_context"} <= trace.keys()
    # first stage is always the classifier
    assert trace["stages"][0]["stage"] == "classifier"


@pytest.mark.asyncio
async def test_api_sdp_ingest_then_visible(tmp_path):
    client = await _client(tmp_path)
    before = client.get("/api/health").json()["memories"]
    r = client.post(
        "/api/memory/sdp_ingest",
        json={
            "user_query": "What database and port do we use?",
            "assistant_reply": "We use PostgreSQL 16 running on port 5432.",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "sdp"
    assert body["count"] >= 1, body
    after = client.get("/api/health").json()["memories"]
    assert after == before + body["count"]


@pytest.mark.asyncio
async def test_api_llm_ingest_unavailable_without_extractor(tmp_path):
    client = await _client(tmp_path)  # enable_sam_llm=False → extractor is None
    r = client.post(
        "/api/memory/ingest",
        json={"user_query": "hi", "assistant_reply": "hello"},
    )
    assert r.status_code == 503, r.text


@pytest.mark.asyncio
async def test_api_recall_stream_emits_stages_then_done(tmp_path):
    import json as _json

    client = await _client(tmp_path)
    frames: list[tuple[str, str]] = []
    with client.stream(
        "POST", "/api/memory/recall/stream", json={"query": "what port", "top_k": 3}
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        cur = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                cur = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                frames.append((cur or "", line.split(":", 1)[1].strip()))

    kinds = [e for e, _ in frames]
    assert "stage" in kinds, kinds  # real per-stage progress
    assert kinds[-1] == "done"
    done = _json.loads(frames[-1][1])
    assert isinstance(done["results"], list)
    assert "formatted_context" in done
    # first emitted stage is always the classifier
    first_stage = _json.loads(next(d for e, d in frames if e == "stage"))
    assert first_stage["stage"] == "classifier"


@pytest.mark.asyncio
async def test_api_delete_and_404(tmp_path):
    client = await _client(tmp_path)
    # create something deletable via SDP
    client.post(
        "/api/memory/sdp_ingest",
        json={
            "user_query": "What language?",
            "assistant_reply": "The backend is written in Python 3.12.",
        },
    )
    listing = client.get("/api/memories?limit=500").json()["memories"]
    target = listing[-1]["id"]
    before = client.get("/api/health").json()["memories"]

    rd = client.delete(f"/api/memories/{target}")
    assert rd.status_code == 200, rd.text
    assert rd.json()["remaining"] == before - 1

    # gone now
    assert client.get(f"/api/memories/{target}").status_code == 404
    assert client.delete(f"/api/memories/{target}").status_code == 404
