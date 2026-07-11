"""End-to-end tests for the relay.

A real uvicorn server runs in a background thread (so SSE streaming works over a
real socket), and an in-test "echo connector" plays the server side. Tests drive
the host side with httpx exactly as a browser extension / native app would.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from mcp_relay.app import create_app
from mcp_relay.registry import RelayState
from mcp_relay.sse import iter_sse

TOKENS = {"alice-token": "alice", "bob-token": "bob"}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def base_url():
    app = create_app(RelayState(dict(TOKENS)))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


# --------------------------------------------------------------------------
# In-test echo connector (the server side of the broker)
# --------------------------------------------------------------------------
def _echo_handle(msg: dict, label: str):
    method = msg.get("method")
    if "id" not in msg or msg.get("id") is None:
        return None  # notification
    mid = msg["id"]
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": f"{label}-server", "version": "0.1.0"},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [{"name": "echo"}]}}
    if method == "tools/call":
        text = (msg.get("params", {}).get("arguments") or {}).get("text", "")
        return {"jsonrpc": "2.0", "id": mid,
                "result": {"content": [{"type": "text", "text": f"{label}: {text}"}]}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "not found"}}


async def _run_echo_connector(base: str, token: str, name: str, label: str, started: asyncio.Event):
    async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(None, connect=10)) as c:
        reg = await c.post("/relay/register", headers={"Authorization": f"Bearer {token}"},
                           json={"server": name})
        reg.raise_for_status()
        s = reg.json()["server_session"]
        hdr = {"X-Relay-Server-Session": s}
        async with c.stream("GET", "/relay/stream", headers=hdr) as resp:
            started.set()
            async for event, data in iter_sse(resp.aiter_lines()):
                if event != "message" or not isinstance(data, dict):
                    continue
                if data.get("type") != "message":
                    continue  # ignore open/close — echo is stateless
                reply = _echo_handle(data["payload"], label)
                if reply is not None:
                    await c.post("/relay/message", headers=hdr,
                                 json={"session": data["session"], "payload": reply})


@contextlib.asynccontextmanager
async def echo_connector(base: str, token: str = "alice-token", name: str = "echo",
                         label: str | None = None, min_connected: int = 1):
    started = asyncio.Event()
    task = asyncio.create_task(_run_echo_connector(base, token, name, label or name, started))
    await asyncio.wait_for(started.wait(), timeout=5)
    # Wait until the relay reports the expected number of servers connected.
    async with httpx.AsyncClient(base_url=base) as probe:
        for _ in range(200):
            if (await probe.get("/health")).json()["stats"]["servers_connected"] >= min_connected:
                break
            await asyncio.sleep(0.02)
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


# --------------------------------------------------------------------------
# Host-side helpers
# --------------------------------------------------------------------------
_counter = 0


def _next_id() -> int:
    global _counter
    _counter += 1
    return _counter


async def rpc(client, method, *, params=None, token="alice-token", session=None, server=None):
    """Send a JSON-RPC request; returns (status, session_id, message_or_errorbody)."""
    msg = {"jsonrpc": "2.0", "id": _next_id(), "method": method}
    if params is not None:
        msg["params"] = params
    url = "/mcp"
    if session is None and server:
        url += f"?server={server}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if session:
        headers["Mcp-Session-Id"] = session
    async with client.stream("POST", url, headers=headers, content=json.dumps(msg)) as resp:
        new_session = resp.headers.get("mcp-session-id")
        if resp.status_code != 200:
            raw = await resp.aread()
            return resp.status_code, new_session, (json.loads(raw) if raw else None)
        async for event, data in iter_sse(resp.aiter_lines()):
            if event == "message":
                return resp.status_code, new_session, data
    return resp.status_code, new_session, None


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
async def test_initialize_list_call(base_url):
    async with echo_connector(base_url), httpx.AsyncClient(base_url=base_url) as client:
        status, session, init = await rpc(client, "initialize", server="echo",
                                          params={"protocolVersion": "2024-11-05",
                                                  "capabilities": {}, "clientInfo": {"name": "t"}})
        assert status == 200
        assert session  # relay assigned a session id
        assert init["result"]["serverInfo"]["name"] == "echo-server"

        status, _, tools = await rpc(client, "tools/list", session=session)
        assert status == 200
        assert [t["name"] for t in tools["result"]["tools"]] == ["echo"]

        status, _, called = await rpc(client, "tools/call", session=session,
                                      params={"name": "echo", "arguments": {"text": "hi"}})
        assert status == 200
        assert called["result"]["content"][0]["text"] == "echo: hi"


async def test_notification_returns_202(base_url):
    async with echo_connector(base_url), httpx.AsyncClient(base_url=base_url) as client:
        _, session, _ = await rpc(client, "initialize", server="echo", params={})
        resp = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer alice-token", "Mcp-Session-Id": session},
            content=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        )
        assert resp.status_code == 202


async def test_unauthorized(base_url):
    async with httpx.AsyncClient(base_url=base_url) as client:
        resp = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer not-a-real-token"},
            content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        )
        assert resp.status_code == 401


async def test_unknown_session_404(base_url):
    async with httpx.AsyncClient(base_url=base_url) as client:
        status, _, body = await rpc(client, "tools/list", session="does-not-exist")
        assert status == 404
        assert body["error"]["code"] == -32001


async def test_user_isolation(base_url):
    # Only alice has a connected 'echo' server. Bob must not be able to reach it.
    async with echo_connector(base_url, token="alice-token", name="echo"), \
            httpx.AsyncClient(base_url=base_url) as client:
        # Bob has no servers at all -> 503 when initializing against 'echo'.
        status, _, body = await rpc(client, "initialize", token="bob-token", server="echo",
                                    params={})
        assert status == 503
        assert body["error"]["code"] == -32002

        # Alice opens a real session...
        _, alice_session, _ = await rpc(client, "initialize", token="alice-token",
                                        server="echo", params={})
        assert alice_session
        # ...and bob cannot hijack it with the session id.
        status, _, body = await rpc(client, "tools/list", token="bob-token",
                                    session=alice_session)
        assert status == 404


async def test_default_server_when_single(base_url):
    # No ?server= given; the user's single registered server is chosen automatically.
    async with echo_connector(base_url), httpx.AsyncClient(base_url=base_url) as client:
        status, session, init = await rpc(client, "initialize", params={})
        assert status == 200
        assert init["result"]["serverInfo"]["name"] == "echo-server"


async def test_aggregate_all_servers(base_url):
    # Two servers under alice; server=* aggregates them with namespaced tools.
    async with echo_connector(base_url, name="echo", label="echo", min_connected=1), \
            echo_connector(base_url, name="calc", label="calc", min_connected=2), \
            httpx.AsyncClient(base_url=base_url) as client:
        status, session, init = await rpc(client, "initialize", server="*", params={})
        assert status == 200
        assert init["result"]["serverInfo"]["name"] == "mcp-relay-aggregate"
        assert "tools" in init["result"]["capabilities"]

        _, _, tools = await rpc(client, "tools/list", session=session)
        names = sorted(t["name"] for t in tools["result"]["tools"])
        assert names == ["calc__echo", "echo__echo"]  # namespaced, no collision

        # Each namespaced call must route to the owning backend.
        _, _, r1 = await rpc(client, "tools/call", session=session,
                             params={"name": "calc__echo", "arguments": {"text": "x"}})
        assert r1["result"]["content"][0]["text"] == "calc: x"
        _, _, r2 = await rpc(client, "tools/call", session=session,
                             params={"name": "echo__echo", "arguments": {"text": "y"}})
        assert r2["result"]["content"][0]["text"] == "echo: y"


async def _read_response(proc, want_id, timeout=10.0):
    """Read JSON lines from the bridge subprocess until the response with want_id."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"no response for id {want_id}")
        raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
        if not raw:
            raise EOFError("bridge closed stdout")
        try:
            msg = json.loads(raw.decode())
        except json.JSONDecodeError:
            continue
        if msg.get("id") == want_id:
            return msg


async def test_stdio_bridge_subprocess(base_url):
    """End-to-end through the actual mcp-relay-client stdio bridge subprocess."""
    import sys

    async with echo_connector(base_url, name="echo", label="echo"):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "mcp_relay.client",
            "--url", base_url, "--token", "alice-token", "--server", "echo",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async def send(obj):
                proc.stdin.write((json.dumps(obj) + "\n").encode())
                await proc.stdin.drain()

            await send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "ide", "version": "1"}}})
            init = await _read_response(proc, 1)
            assert init["result"]["serverInfo"]["name"] == "echo-server"

            await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "echo", "arguments": {"text": "via bridge"}}})
            called = await _read_response(proc, 2)
            assert called["result"]["content"][0]["text"] == "echo: via bridge"
        finally:
            proc.stdin.close()
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(proc.wait(), timeout=5)
