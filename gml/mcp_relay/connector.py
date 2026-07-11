"""Connector: bridges a local stdio MCP server into the relay.

Because the relay is a broker and the MCP server lives behind NAT / on a user's
machine, the server can't be dialed directly. This connector dials *out* to the
relay, opens a downlink SSE stream, and for every host session the relay reports
it launches a dedicated subprocess of the backing MCP server. Messages flow:

    relay --downlink(SSE)--> connector --stdin--> server subprocess
    server subprocess --stdout--> connector --uplink(POST)--> relay

One subprocess per host session keeps JSON-RPC ids isolated end to end.

Usage:
    mcp-relay-connector --url http://127.0.0.1:8080 --token <user-token> \\
        --name echo -- python examples/echo_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import httpx

from .sse import iter_sse

log = logging.getLogger("mcp-relay.connector")

# Children that see no traffic for this long are presumed abandoned: the
# relay's 'close' envelope is best-effort (lost on broker restarts and
# downlink drops), so without a timeout dead sessions pile up forever —
# one full MCP server process each.
DEFAULT_IDLE_TIMEOUT = 900.0
DEFAULT_MAX_SESSIONS = 8


class Connector:
    def __init__(
        self,
        url: str,
        token: str,
        name: str,
        command: list[str],
        info: dict,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.name = name
        self.command = command
        self.info = info
        self.idle_timeout = idle_timeout
        self.max_sessions = max_sessions
        self.server_session: Optional[str] = None
        self.sessions: dict[str, asyncio.subprocess.Process] = {}
        self.last_active: dict[str, float] = {}
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
        self._reaper: Optional[asyncio.Task] = None

    # -- lifecycle ----------------------------------------------------------
    async def run(self) -> None:
        backoff = 1.0
        if self.idle_timeout > 0:
            self._reaper = asyncio.create_task(self._reap_idle_loop())
        try:
            while True:
                try:
                    if not self.server_session:
                        await self.register()
                    await self.stream()  # blocks until the downlink drops
                    backoff = 1.0
                except httpx.HTTPStatusError as exc:
                    if exc.response is not None and exc.response.status_code == 401:
                        log.warning("relay rejected session; re-registering")
                        self.server_session = None
                    else:
                        log.warning("relay error: %s", exc)
                except (httpx.HTTPError, ConnectionError) as exc:
                    log.warning("connection lost: %s", exc)
                log.info("reconnecting in %.0fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        for sid in list(self.sessions):
            await self.close_session(sid)
        await self.client.aclose()

    async def _reap_idle_loop(self) -> None:
        interval = max(1.0, min(60.0, self.idle_timeout / 4))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for sid in list(self.sessions):
                idle = now - self.last_active.get(sid, now)
                if idle > self.idle_timeout:
                    log.info("session %s idle %.0fs > %.0fs — reaping",
                             sid, idle, self.idle_timeout)
                    await self.close_session(sid)

    # -- relay legs ---------------------------------------------------------
    async def register(self) -> None:
        resp = await self.client.post(
            f"{self.url}/relay/register",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"server": self.name, "info": self.info, "global": bool(os.environ.get("RELAY_REGISTER_GLOBAL"))},
        )
        resp.raise_for_status()
        self.server_session = resp.json()["server_session"]
        log.info("registered server %r as user %r", self.name, resp.json().get("user"))
        # A fresh server_session means the relay has no record of our old host
        # sessions — it will never route to them or send their 'close'. Kill
        # the orphaned children now or they linger forever.
        if self.sessions:
            log.info("re-registered; closing %d stale session(s)", len(self.sessions))
            for sid in list(self.sessions):
                await self.close_session(sid)

    async def stream(self) -> None:
        headers = {"X-Relay-Server-Session": self.server_session or ""}
        async with self.client.stream("GET", f"{self.url}/relay/stream", headers=headers) as resp:
            if resp.status_code != 200:
                await resp.aread()
                resp.raise_for_status()
            log.info("downlink open")
            async for event, data in iter_sse(resp.aiter_lines()):
                if event == "message" and isinstance(data, dict):
                    await self.dispatch(data)

    async def uplink(self, sid: str, payload: dict) -> None:
        try:
            await self.client.post(
                f"{self.url}/relay/message",
                headers={"X-Relay-Server-Session": self.server_session or ""},
                json={"session": sid, "payload": payload},
            )
        except httpx.HTTPError as exc:
            log.warning("uplink failed for session %s: %s", sid, exc)

    # -- envelope handling --------------------------------------------------
    async def dispatch(self, env: dict) -> None:
        etype = env.get("type")
        sid = env.get("session")
        if not sid:
            return
        if etype == "open":
            await self.open_session(sid, env.get("user"))
        elif etype == "message":
            await self.feed(sid, env.get("payload"), env.get("user"))
        elif etype == "close":
            await self.close_session(sid)

    async def open_session(self, sid: str, user: str | None = None) -> None:
        if sid in self.sessions:
            return
        # Hard cap on live children — evict the least-recently-active session
        # to make room. Backstop against a flood of opens outpacing the idle
        # reaper.
        if self.max_sessions > 0:
            while len(self.sessions) >= self.max_sessions:
                lru = min(
                    self.sessions,
                    key=lambda s: self.last_active.get(s, 0.0),
                )
                log.info("session cap %d reached — evicting %s", self.max_sessions, lru)
                await self.close_session(lru)
        # Per-user scoping: the relay forwards the authenticated user_id in the
        # 'open' envelope. Pass it to the spawned MCP server as GML_MCP_USER so
        # each host session reads/writes only that tenant's memory.
        senv = os.environ.copy()
        if user:
            senv["GML_MCP_USER"] = user
        proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit, so server logs surface in the connector console
            env=senv,
        )
        self.sessions[sid] = proc
        self.last_active[sid] = time.monotonic()
        log.info("session %s opened (pid %s)", sid, proc.pid)
        asyncio.create_task(self._pump_stdout(sid, proc))

    async def feed(self, sid: str, payload: Any, user: str | None = None) -> None:
        if not isinstance(payload, dict):
            return
        proc = self.sessions.get(sid)
        if proc is None:
            # Tolerate a missing 'open' (e.g. this connector restarted and the
            # relay still holds the session). The relay stamps "user" on every
            # envelope so the re-opened child keeps the right tenant scope.
            await self.open_session(sid, user)
            proc = self.sessions.get(sid)
        if proc is None or proc.stdin is None:
            return
        self.last_active[sid] = time.monotonic()
        try:
            proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            await self.close_session(sid)

    async def _pump_stdout(self, sid: str, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                self.last_active[sid] = time.monotonic()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("session %s emitted non-JSON: %s", sid, line[:200])
                    continue
                await self.uplink(sid, payload)
        finally:
            log.info("session %s stdout closed", sid)
            self.sessions.pop(sid, None)
            self.last_active.pop(sid, None)
            await self._reap(proc)

    async def close_session(self, sid: str) -> None:
        proc = self.sessions.pop(sid, None)
        self.last_active.pop(sid, None)
        if proc is None:
            return
        log.info("session %s closing", sid)
        await self._reap(proc)

    @staticmethod
    async def _reap(proc: asyncio.subprocess.Process) -> None:
        """Make sure a child is dead AND waited on (no zombies, no leaks)."""
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.warning("pid %s did not exit after SIGKILL", proc.pid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcp-relay-connector",
        description="Bridge a stdio MCP server into the relay.",
        epilog="Put the server command after '--', e.g. -- python examples/echo_server.py",
    )
    parser.add_argument("--url", default=os.environ.get("RELAY_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--token", default=os.environ.get("RELAY_TOKEN"),
                        help="user bearer token (or RELAY_TOKEN env)")
    parser.add_argument("--name", default="default", help="server name to register")
    parser.add_argument("--info", default="{}", help="JSON server info/metadata")
    parser.add_argument(
        "--idle-timeout", type=float,
        default=float(os.environ.get("RELAY_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)),
        help="kill child sessions with no traffic for this many seconds "
             "(0 disables; env RELAY_IDLE_TIMEOUT)",
    )
    parser.add_argument(
        "--max-sessions", type=int,
        default=int(os.environ.get("RELAY_MAX_SESSIONS", DEFAULT_MAX_SESSIONS)),
        help="max concurrent child sessions; least-recently-active is evicted "
             "(0 = unlimited; env RELAY_MAX_SESSIONS)",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the stdio MCP server command (after '--')")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.token:
        parser.error("a --token (or RELAY_TOKEN) is required")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no server command given (put it after '--')")
    try:
        info = json.loads(args.info)
    except json.JSONDecodeError:
        parser.error("--info must be valid JSON")

    connector = Connector(
        args.url, args.token, args.name, command, info,
        idle_timeout=args.idle_timeout, max_sessions=args.max_sessions,
    )
    try:
        asyncio.run(connector.run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
