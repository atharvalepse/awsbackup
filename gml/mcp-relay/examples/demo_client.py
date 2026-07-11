#!/usr/bin/env python3
"""A tiny MCP host that talks to the relay over Streamable HTTP.

Run the relay and a connector first (see README), then:
    python examples/demo_client.py --url http://127.0.0.1:8080 --token <token> --server echo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

# Allow running straight from the repo without installing the package.
sys.path.insert(0, __file__.rsplit("/examples/", 1)[0] + "/src")
from mcp_relay.sse import iter_sse  # noqa: E402


class RelayClient:
    def __init__(self, url: str, token: str, server: str | None):
        self.url = url.rstrip("/")
        self.token = token
        self.server = server
        self.session_id: str | None = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._next_id = 0

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def call(self, method: str, params: dict | None = None) -> dict:
        msg = {"jsonrpc": "2.0", "id": self._id(), "method": method}
        if params is not None:
            msg["params"] = params
        params_url = f"{self.url}/mcp"
        if self.session_id is None and self.server:
            params_url += f"?server={self.server}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        async with self.client.stream("POST", params_url, headers=headers,
                                       content=json.dumps(msg)) as resp:
            if resp.headers.get("mcp-session-id"):
                self.session_id = resp.headers["mcp-session-id"]
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"HTTP {resp.status_code}: {body.decode()}")
            async for event, data in iter_sse(resp.aiter_lines()):
                if event == "message":
                    return data
        raise RuntimeError("no response from relay")

    async def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        await self.client.post(f"{self.url}/mcp", headers=headers, content=json.dumps(msg))

    async def close(self) -> None:
        await self.client.aclose()


async def main_async(args) -> None:
    c = RelayClient(args.url, args.token, args.server)
    try:
        init = await c.call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "demo-client", "version": "0.1.0"},
        })
        print("initialize ->", json.dumps(init["result"]["serverInfo"]))
        await c.notify("notifications/initialized")

        tools = await c.call("tools/list")
        names = [t["name"] for t in tools["result"]["tools"]]
        print("tools/list ->", names)

        result = await c.call("tools/call", {"name": "echo", "arguments": {"text": args.text}})
        print("tools/call ->", result["result"]["content"][0]["text"])
    finally:
        await c.close()


def main() -> int:
    p = argparse.ArgumentParser(prog="demo_client")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--token", required=True)
    p.add_argument("--server", default=None, help="target server name (optional if user has one)")
    p.add_argument("--text", default="hello relay")
    args = p.parse_args()
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
