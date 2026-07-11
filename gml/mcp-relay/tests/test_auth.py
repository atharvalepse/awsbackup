"""Auth tests.

Pure-function tests (passwords, sessions) always run. The Postgres-backed tests
run only if a reachable test database is configured via RELAY_TEST_DATABASE_URL
(default: ``postgresql:///mcp_relay_test``); otherwise they are skipped.
"""

import asyncio
import os
import socket
import threading
import time

import httpx
import pytest
import pytest_asyncio
import uvicorn

from mcp_relay.app import create_app
from mcp_relay.auth import (
    DatabaseAuth,
    EmailTaken,
    hash_password,
    make_session,
    read_session,
    verify_password,
)
from mcp_relay.registry import RelayState

TEST_DSN = os.environ.get("RELAY_TEST_DATABASE_URL", "postgresql:///mcp_relay_test")


# -- pure functions ---------------------------------------------------------
def test_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert h.startswith("pbkdf2_sha256$")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong", h)


def test_session_roundtrip():
    secret = "topsecret"
    tok = make_session("42", secret, ttl=60)
    assert read_session(tok, secret) == "42"
    assert read_session(tok, "other-secret") is None          # bad signature
    assert read_session(make_session("7", secret, ttl=-1), secret) is None  # expired


# -- Postgres-backed --------------------------------------------------------
async def _open_or_skip(auth: DatabaseAuth) -> None:
    try:
        await auth.pool.open(wait=True, timeout=3)
    except Exception:
        await auth.pool.close()
        pytest.skip(f"no test Postgres at {TEST_DSN}")


@pytest_asyncio.fixture
async def db():
    auth = DatabaseAuth(TEST_DSN, "session-secret")
    await _open_or_skip(auth)
    await auth.init_schema()
    async with auth.pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("TRUNCATE users, api_tokens RESTART IDENTITY CASCADE")
    yield auth
    await auth.shutdown()


@pytest.fixture
def base_url_db():
    # Prep schema + clean tables (skips the whole test if no DB).
    async def prep():
        auth = DatabaseAuth(TEST_DSN, "session-secret")
        await _open_or_skip(auth)
        await auth.init_schema()
        async with auth.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("TRUNCATE users, api_tokens RESTART IDENTITY CASCADE")
        await auth.shutdown()

    asyncio.run(prep())

    auth = DatabaseAuth(TEST_DSN, "session-secret")
    app = create_app(RelayState(auth), auth_store=auth, session_secret="session-secret")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


async def test_signup_login_token_lifecycle(db):
    uid = await db.create_user("Alice@Example.com", "supersecret")
    assert uid

    with pytest.raises(EmailTaken):
        await db.create_user("alice@example.com", "another")  # case-insensitive unique

    assert await db.verify_login("alice@example.com", "supersecret") == uid
    assert await db.verify_login("alice@example.com", "wrong") is None
    assert await db.verify_login("nobody@example.com", "x") is None

    created = await db.create_token(uid, "cursor-laptop")
    raw = created["token"]
    assert await db.user_for_token(raw) == uid          # resolves to the user
    assert await db.user_for_token("not-a-token") is None

    toks = await db.list_tokens(uid)
    assert len(toks) == 1 and toks[0]["label"] == "cursor-laptop" and not toks[0]["revoked"]

    assert await db.revoke_token(uid, created["id"]) is True
    assert await db.user_for_token(raw) is None          # cache evicted on revoke
    assert (await db.list_tokens(uid))[0]["revoked"] is True


async def test_http_signup_to_relay_flow(base_url_db):
    """Signup over HTTP, issue a token, then use it to register+initialize."""
    async with httpx.AsyncClient(base_url=base_url_db) as c:
        r = await c.post("/auth/signup", json={"email": "bob@example.com", "password": "supersecret"})
        assert r.status_code == 201
        session = r.json()["session"]

        r = await c.post("/auth/tokens", headers={"Authorization": f"Bearer {session}"},
                         json={"label": "ide"})
        assert r.status_code == 201
        token = r.json()["token"]

        # The issued token now works as a relay bearer token.
        reg = await c.post("/relay/register", headers={"Authorization": f"Bearer {token}"},
                           json={"server": "echo"})
        assert reg.status_code == 200

        # And a bogus token is rejected on the host endpoint.
        bad = await c.post("/mcp", headers={"Authorization": "Bearer nope"},
                           content='{"jsonrpc":"2.0","id":1,"method":"initialize"}')
        assert bad.status_code == 401
