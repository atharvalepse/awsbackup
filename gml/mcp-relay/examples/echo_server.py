#!/usr/bin/env python3
"""A minimal stdio MCP server used to demo/test the relay.

It speaks newline-delimited JSON-RPC on stdin/stdout (the same framing the
connector bridges). It exposes one tool, ``echo``, plus ``initialize`` /
``tools/list`` / ``ping``. This is intentionally tiny — it is not a full MCP
implementation, just enough to exercise routing end-to-end.
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2024-11-05"


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")

    # Notifications (no id) — acknowledge silently.
    if "id" not in msg or msg_id is None:
        return None

    if method == "initialize":
        return _ok(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo-server", "version": "0.1.0"},
        })
    if method == "ping":
        return _ok(msg_id, {})
    if method == "tools/list":
        return _ok(msg_id, {"tools": [{
            "name": "echo",
            "description": "Echo back the provided text.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }]})
    if method == "tools/call":
        params = msg.get("params") or {}
        if params.get("name") != "echo":
            return _err(msg_id, -32602, f"unknown tool {params.get('name')!r}")
        text = (params.get("arguments") or {}).get("text", "")
        return _ok(msg_id, {"content": [{"type": "text", "text": f"echo: {text}"}], "isError": False})

    return _err(msg_id, -32601, f"method not found: {method}")


def _ok(msg_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = handle(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
