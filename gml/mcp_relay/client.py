"""stdio <-> relay bridge: the "MCP definition" an IDE launches.

IDEs (Claude Desktop, Cursor, VS Code, Windsurf, ...) start an MCP server as a
stdio command. This bridge is that command: it reads newline-delimited JSON-RPC
on stdin and forwards it to the relay's Streamable HTTP `/mcp` endpoint, writing
replies back on stdout. It also opens the relay's standby SSE stream so
server-initiated messages (notifications, sampling requests) reach the IDE.

Target selection:
    --server NAME   reach exactly that one server
    --all           reach the aggregated view of all the user's servers (server=*)
    (neither)       the user's single server, if they have exactly one

Example IDE config:
    {
      "command": "mcp-relay-client",
      "args": ["--url", "https://relay.example.com", "--all"],
      "env": { "RELAY_TOKEN": "..." }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from typing import Optional

import httpx

from .sse import iter_sse


class Bridge:
    def __init__(self, url: str, token: str, server: Optional[str]):
        self.url = url.rstrip("/")
        self.token = token
        self.server = server  # None, a name, or "*"
        self.session_id: Optional[str] = None
        self._session_ready = asyncio.Event()
        self._stdout_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))

    # -- plumbing -----------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _mcp_url(self) -> str:
        # The ?server= selector is only meaningful on the opening (no session yet).
        if self.session_id is None and self.server:
            return f"{self.url}/mcp?server={self.server}"
        return f"{self.url}/mcp"

    async def _write(self, obj: dict) -> None:
        line = json.dumps(obj) + "\n"
        async with self._stdout_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    # -- message handling ---------------------------------------------------
    async def handle(self, msg: dict) -> None:
        is_request = "method" in msg and msg.get("id") is not None
        if not is_request:
            await self._post_no_reply(msg)  # notification, or response to a server request
            return
        try:
            reply = await self._request(msg)
        except Exception as exc:  # surface as a JSON-RPC error rather than crashing the IDE
            reply = {"jsonrpc": "2.0", "id": msg.get("id"),
                     "error": {"code": -32099, "message": f"relay bridge error: {exc}"}}
        if reply is not None:
            await self._write(reply)

    async def _request(self, msg: dict) -> Optional[dict]:
        async with self.client.stream("POST", self._mcp_url(), headers=self._headers(),
                                       content=json.dumps(msg)) as resp:
            sid = resp.headers.get("mcp-session-id")
            if sid and not self.session_id:
                self.session_id = sid
                self._session_ready.set()
            if resp.status_code == 202:
                return None
            if resp.status_code != 200:
                body = await resp.aread()
                return {"jsonrpc": "2.0", "id": msg.get("id"),
                        "error": {"code": -32098,
                                  "message": f"relay HTTP {resp.status_code}: {body.decode()[:300]}"}}
            async for event, data in iter_sse(resp.aiter_lines()):
                if event == "message":
                    return data
        return None

    async def _post_no_reply(self, msg: dict) -> None:
        if self.session_id is None and "method" in msg:
            # wait briefly for the session to come up (shouldn't normally happen)
            try:
                await asyncio.wait_for(self._session_ready.wait(), timeout=10)
            except asyncio.TimeoutError:
                return
        resp = await self.client.post(self._mcp_url(), headers=self._headers(),
                                      content=json.dumps(msg))
        sid = resp.headers.get("mcp-session-id")
        if sid and not self.session_id:
            self.session_id = sid
            self._session_ready.set()

    # -- standby stream (server-initiated traffic) -------------------------
    async def _standby(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with self.client.stream("GET", f"{self.url}/mcp",
                                               headers=self._headers()) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                        continue
                    backoff = 1.0
                    async for event, data in iter_sse(resp.aiter_lines()):
                        if event == "message":
                            await self._write(data)
            except httpx.HTTPError:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    # -- main loop ----------------------------------------------------------
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def reader():  # stdin reads are blocking; do them on a thread
            for line in sys.stdin:
                loop.call_soon_threadsafe(queue.put_nowait, line)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # EOF

        threading.Thread(target=reader, daemon=True).start()

        standby_task: Optional[asyncio.Task] = None
        try:
            while True:
                line = await queue.get()
                if line is None:  # stdin closed
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Establish the session synchronously on initialize, then fan out.
                if msg.get("method") == "initialize" and self.session_id is None:
                    await self.handle(msg)
                    if self.session_id and standby_task is None:
                        standby_task = asyncio.create_task(self._standby())
                else:
                    asyncio.create_task(self.handle(msg))
        finally:
            if standby_task:
                standby_task.cancel()
            await self.client.aclose()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mcp-relay-client",
        description="Bridge a stdio MCP client (IDE) to a remote mcp-relay.",
    )
    p.add_argument("--url", default=os.environ.get("RELAY_URL", "http://127.0.0.1:8080"))
    p.add_argument("--token", default=os.environ.get("RELAY_TOKEN"),
                   help="user bearer token (or RELAY_TOKEN env)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--server", help="target a single server by name")
    group.add_argument("--all", action="store_true",
                       help="aggregate all of the user's servers (server=*)")
    args = p.parse_args(argv)

    if not args.token:
        p.error("a --token (or RELAY_TOKEN) is required")
    server = "*" if args.all else args.server

    bridge = Bridge(args.url, args.token, server)
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
