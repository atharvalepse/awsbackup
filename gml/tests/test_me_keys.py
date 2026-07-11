"""Tests for the self-service /api/me/keys endpoints.

These cover the per-user key surface a logged-in user manages for themselves
(the gml_ bearer keys the Claude Desktop extension uses), distinct from the
master-key-only /api/admin/keys endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from orchestration.auth.tokens import make_access_token
from orchestration.embedder import StubEmbedder
from orchestration.server import build_app, build_default_state


async def _build_client(tmp_path, monkeypatch) -> TestClient:
    # GML_API_KEY enables auth; GML_USER_KEYS_FILE isolates the key store to a
    # tmp file so tests don't touch ~/.gml/users.jsonl.
    monkeypatch.setenv("GML_API_KEY", "test-master-key")
    monkeypatch.setenv("GML_USER_KEYS_FILE", str(tmp_path / "users.jsonl"))
    monkeypatch.delenv("GML_STORAGE_BACKEND", raising=False)
    state = await build_default_state(
        embedder=StubEmbedder(dim=384),
        memory_path=tmp_path / "memories.jsonl",
        enable_sam_llm=False,
        stub_client=True,
    )
    return TestClient(build_app(state))


def _bearer(user_id: str) -> dict:
    """A user JWT (same shape the auth middleware decodes from /auth/login)."""
    return {"Authorization": f"Bearer {make_access_token(user_id)['access_token']}"}


@pytest.mark.asyncio
async def test_me_issue_returns_gml_prefixed_key(tmp_path, monkeypatch):
    client = await _build_client(tmp_path, monkeypatch)
    r = client.post("/api/me/keys", json={"label": "laptop"}, headers=_bearer("usr_a"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"].startswith("gml_")
    assert body["user_id"] == "usr_a"
    assert body["label"] == "laptop"


@pytest.mark.asyncio
async def test_me_list_shows_only_own_keys(tmp_path, monkeypatch):
    client = await _build_client(tmp_path, monkeypatch)
    client.post("/api/me/keys", json={"label": "a1"}, headers=_bearer("usr_a"))
    client.post("/api/me/keys", json={"label": "a2"}, headers=_bearer("usr_a"))
    client.post("/api/me/keys", json={"label": "b1"}, headers=_bearer("usr_b"))

    r = client.get("/api/me/keys", headers=_bearer("usr_a"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert all(k["user_id"] == "usr_a" for k in body["keys"])
    assert sorted(k["label"] for k in body["keys"]) == ["a1", "a2"]
    # Listings are redacted: a preview, never the full secret.
    assert all("key" not in k and "key_preview" in k for k in body["keys"])


@pytest.mark.asyncio
async def test_me_revoke_rejects_another_users_key(tmp_path, monkeypatch):
    client = await _build_client(tmp_path, monkeypatch)
    b_key = client.post(
        "/api/me/keys", json={"label": "b"}, headers=_bearer("usr_b")
    ).json()["key"]

    # usr_a must NOT be able to revoke usr_b's key.
    r = client.delete("/api/me/keys", params={"key": b_key}, headers=_bearer("usr_a"))
    assert r.status_code == 403, r.text

    # The key is untouched — usr_b still sees it.
    r2 = client.get("/api/me/keys", headers=_bearer("usr_b"))
    assert any(k["label"] == "b" for k in r2.json()["keys"])

    # usr_b CAN revoke their own key.
    r3 = client.delete("/api/me/keys", params={"key": b_key}, headers=_bearer("usr_b"))
    assert r3.status_code == 200, r3.text
    assert r3.json()["revoked"] is True


@pytest.mark.asyncio
async def test_me_revoke_by_preview_works_for_own_key(tmp_path, monkeypatch):
    # The listing only exposes key_preview, so revoke must also accept it
    # (scoped to the caller's own keys).
    client = await _build_client(tmp_path, monkeypatch)
    client.post("/api/me/keys", json={"label": "old"}, headers=_bearer("usr_a"))
    preview = client.get("/api/me/keys", headers=_bearer("usr_a")).json()["keys"][0][
        "key_preview"
    ]

    r = client.delete("/api/me/keys", params={"key": preview}, headers=_bearer("usr_a"))
    assert r.status_code == 200, r.text
    assert r.json()["revoked"] is True
    assert client.get("/api/me/keys", headers=_bearer("usr_a")).json()["count"] == 0


@pytest.mark.asyncio
async def test_me_keys_rejects_master_caller(tmp_path, monkeypatch):
    # The master/admin caller is not a "user" here — must use /api/admin/keys.
    client = await _build_client(tmp_path, monkeypatch)
    r = client.get("/api/me/keys", headers={"Authorization": "Bearer test-master-key"})
    assert r.status_code == 403, r.text
