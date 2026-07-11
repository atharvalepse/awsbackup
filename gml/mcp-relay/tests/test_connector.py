"""Unit tests for the Connector's child-process lifecycle.

These drive the Connector class directly with a trivial echo child — no relay
broker needed. They pin the leak fixes: idle reaping, the LRU session cap,
close-all on re-register, and zombie-free shutdown.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from mcp_relay.connector import Connector

# A child that echoes one JSON line per input line and exits on EOF.
ECHO_CHILD = [
    sys.executable, "-u", "-c",
    "import sys\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line)\n"
    "    sys.stdout.flush()\n",
]


def make_connector(**kwargs) -> Connector:
    c = Connector(
        url="http://127.0.0.1:1",  # never dialed in these tests
        token="t",
        name="test",
        command=ECHO_CHILD,
        info={},
        **kwargs,
    )
    # Don't let uplink try the network: collect payloads instead.
    c.uplinked = []

    async def fake_uplink(sid, payload):
        c.uplinked.append((sid, payload))

    c.uplink = fake_uplink
    return c


async def wait_for(predicate, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_open_feed_close_reaps_child():
    c = make_connector()
    await c.open_session("s1")
    proc = c.sessions["s1"]
    await c.feed("s1", {"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert await wait_for(lambda: len(c.uplinked) == 1)
    await c.close_session("s1")
    assert "s1" not in c.sessions
    assert "s1" not in c.last_active
    assert proc.returncode is not None  # dead AND waited on


@pytest.mark.asyncio
async def test_idle_sessions_are_reaped():
    c = make_connector(idle_timeout=0.2)
    # Start the reaper loop the way run() would, without dialing the relay.
    c._reaper = asyncio.create_task(c._reap_idle_loop())
    try:
        await c.open_session("s1")
        proc = c.sessions["s1"]
        assert await wait_for(lambda: "s1" not in c.sessions, timeout=5.0), \
            "idle session was never reaped"
        assert await wait_for(lambda: proc.returncode is not None)
    finally:
        await c.shutdown()


@pytest.mark.asyncio
async def test_activity_defers_idle_reaping():
    c = make_connector(idle_timeout=0.6)
    c._reaper = asyncio.create_task(c._reap_idle_loop())
    try:
        await c.open_session("s1")
        # Keep it busy for ~1s — longer than the idle timeout.
        for i in range(5):
            await asyncio.sleep(0.2)
            await c.feed("s1", {"jsonrpc": "2.0", "id": i, "method": "ping"})
        assert "s1" in c.sessions, "active session must not be reaped"
    finally:
        await c.shutdown()


@pytest.mark.asyncio
async def test_session_cap_evicts_lru():
    c = make_connector(max_sessions=2)
    await c.open_session("old")
    await asyncio.sleep(0.05)
    await c.open_session("mid")
    # Touch "old" so "mid" becomes the least-recently-active.
    await c.feed("old", {"jsonrpc": "2.0", "id": 1, "method": "ping"})
    await c.open_session("new")
    assert set(c.sessions) == {"old", "new"}
    await c.shutdown()


@pytest.mark.asyncio
async def test_reregister_closes_stale_sessions(monkeypatch):
    c = make_connector()
    await c.open_session("stale1")
    await c.open_session("stale2")
    procs = list(c.sessions.values())

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"server_session": "new-ss", "user": "alice"}

    async def fake_post(*a, **k):
        return FakeResp()

    monkeypatch.setattr(c.client, "post", fake_post)
    await c.register()
    assert c.server_session == "new-ss"
    assert not c.sessions, "old children must die on re-register"
    for p in procs:
        assert p.returncode is not None
    await c.shutdown()


@pytest.mark.asyncio
async def test_child_exit_cleans_up_session():
    c = make_connector()
    await c.open_session("s1")
    proc = c.sessions["s1"]
    proc.stdin.close()  # EOF → child exits on its own
    assert await wait_for(lambda: "s1" not in c.sessions)
    assert await wait_for(lambda: proc.returncode is not None)
    await c.shutdown()
