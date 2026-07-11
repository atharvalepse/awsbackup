"""Smoke tests for the gml serve FastAPI surface."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestration.embedder import StubEmbedder
from orchestration.server import build_app, build_default_state


@pytest.mark.asyncio
async def test_server_health_and_chat(tmp_path):
    state = await build_default_state(
        embedder=StubEmbedder(dim=384),
        memory_path=tmp_path / "memories.jsonl",
        enable_sam_llm=False,
        stub_client=True,
    )
    app = build_app(state)
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["memories_loaded"] > 0  # seeded fixture

    # Root
    r = client.get("/")
    assert r.status_code == 200
    assert "POST /chat" in str(r.json())

    # Chat (stub client → canned reply)
    r = client.post("/chat", json={"text": "hello", "target": "deepseek"})
    assert r.status_code == 200, r.text
    chat_body = r.json()
    assert chat_body["session_id"]
    assert chat_body["text"] == "stub response"
    assert chat_body["target_family"] == "deepseek"

    # Add a memory
    r = client.post("/memories", json={
        "content": "manual fact: gravity is real",
        "source": "test",
        "authority_score": 1.0,
    })
    assert r.status_code == 200, r.text
    assert r.json()["id"].startswith("manual-")

    # List memories
    r = client.get("/memories")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


@pytest.mark.asyncio
async def test_server_unknown_target_returns_400(tmp_path):
    state = await build_default_state(
        embedder=StubEmbedder(dim=384),
        memory_path=tmp_path / "memories.jsonl",
        enable_sam_llm=False,
        stub_client=True,
    )
    app = build_app(state)
    client = TestClient(app)
    r = client.post("/chat", json={"text": "hi", "target": "nonexistent"})
    assert r.status_code == 400
    assert "Unknown target" in r.text


@pytest.mark.asyncio
async def test_server_sessions_persist_across_turns(tmp_path):
    state = await build_default_state(
        embedder=StubEmbedder(dim=384),
        memory_path=tmp_path / "memories.jsonl",
        enable_sam_llm=False,
        stub_client=True,
    )
    app = build_app(state)
    client = TestClient(app)

    r1 = client.post("/chat", json={"text": "first", "target": "deepseek"})
    sid = r1.json()["session_id"]
    r2 = client.post("/chat", json={"text": "second", "target": "deepseek", "session_id": sid})
    assert r2.json()["session_id"] == sid

    r3 = client.get("/sessions")
    sessions = r3.json()["sessions"]
    matching = [s for s in sessions if s["session_id"] == sid]
    assert matching and matching[0]["turns"] == 2
