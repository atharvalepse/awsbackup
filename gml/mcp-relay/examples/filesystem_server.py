#!/usr/bin/env python3
"""A real filesystem MCP server (stdio), sandboxed to a root directory.

Speaks newline-delimited JSON-RPC on stdin/stdout — the same framing the relay
connector bridges. Every path in a tool call is resolved relative to <root> and
may not escape it (no path traversal).

Usage (directly, or as the connector's target command):
    python examples/filesystem_server.py /path/to/root

    mcp-relay-connector --url ... --token ... --name filesystem \\
        -- python examples/filesystem_server.py /path/to/root
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"
MAX_READ_BYTES = 1_000_000  # refuse to inline files larger than ~1 MB


class SandboxError(ValueError):
    """Raised when a requested path would escape the sandbox root."""


class FileSystem:
    """All operations are confined to ``root``; results are human-readable text."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel: str | None) -> Path:
        target = (self.root / (rel or ".")).resolve()
        if target != self.root and self.root not in target.parents:
            raise SandboxError(f"path {rel!r} escapes the sandbox root")
        return target

    def _rel(self, p: Path) -> str:
        return "." if p == self.root else str(p.relative_to(self.root))

    # -- tools --------------------------------------------------------------
    def list_directory(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(f"not a directory: {self._rel(p)}")
        lines = []
        for entry in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
            if entry.is_dir():
                lines.append(f"[DIR]  {entry.name}/")
            else:
                lines.append(f"[FILE] {entry.name} ({entry.stat().st_size} bytes)")
        return "\n".join(lines) if lines else "(empty directory)"

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise FileNotFoundError(f"not a file: {self._rel(p)}")
        size = p.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"file too large to read inline ({size} bytes)")
        return p.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {self._rel(p)}"

    def create_directory(self, path: str) -> str:
        p = self._resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"Created directory {self._rel(p)}"

    def move_file(self, source: str, destination: str) -> str:
        src = self._resolve(source)
        dst = self._resolve(destination)
        if not src.exists():
            raise FileNotFoundError(f"no such path: {self._rel(src)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"Moved {self._rel(src)} -> {self._rel(dst)}"

    def delete_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise FileNotFoundError(f"not a file: {self._rel(p)}")
        p.unlink()
        return f"Deleted {self._rel(p)}"

    def get_file_info(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"no such path: {self._rel(p)}")
        st = p.stat()
        info = {
            "path": self._rel(p),
            "type": "directory" if p.is_dir() else "file",
            "size": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "mode": oct(st.st_mode & 0o777),
        }
        return json.dumps(info, indent=2)

    def search_files(self, pattern: str, path: str = ".") -> str:
        base = self._resolve(path)
        matches = []
        for dirpath, dirnames, filenames in os.walk(base):
            for name in dirnames + filenames:
                if fnmatch.fnmatch(name, pattern):
                    matches.append(self._rel(Path(dirpath) / name))
        return "\n".join(sorted(matches)) if matches else f"No matches for {pattern!r}"


TOOLS = [
    {"name": "list_directory", "description": "List entries in a directory.",
     "inputSchema": {"type": "object",
                     "properties": {"path": {"type": "string", "description": "dir, default '.'"}}}},
    {"name": "read_file", "description": "Read a text file's contents.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}},
    {"name": "write_file", "description": "Write text to a file (creates parents).",
     "inputSchema": {"type": "object",
                     "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                     "required": ["path", "content"]}},
    {"name": "create_directory", "description": "Create a directory (mkdir -p).",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}},
    {"name": "move_file", "description": "Move/rename a file or directory.",
     "inputSchema": {"type": "object",
                     "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
                     "required": ["source", "destination"]}},
    {"name": "delete_file", "description": "Delete a file.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}},
    {"name": "get_file_info", "description": "Stat a path (type, size, mtime, mode).",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}},
    {"name": "search_files", "description": "Recursively find names matching a glob pattern.",
     "inputSchema": {"type": "object",
                     "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                     "required": ["pattern"]}},
]


def handle(msg: dict, fs: FileSystem) -> dict | None:
    method = msg.get("method")
    if "id" not in msg or msg.get("id") is None:
        return None  # notification
    mid = msg["id"]

    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "filesystem-server", "version": "0.1.0"},
        })
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = getattr(fs, name, None) if name in {t["name"] for t in TOOLS} else None
        if fn is None:
            return _err(mid, -32602, f"unknown tool {name!r}")
        try:
            text = fn(**args)
            return _ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})
        except TypeError as exc:  # bad/missing arguments
            return _ok(mid, {"content": [{"type": "text", "text": f"bad arguments: {exc}"}],
                             "isError": True})
        except Exception as exc:  # operation failed (not found, sandbox, etc.)
            return _ok(mid, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                             "isError": True})

    return _err(mid, -32601, f"method not found: {method}")


def _ok(mid, result) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else os.getcwd()
    fs = FileSystem(root)
    print(f"filesystem-server sandboxed at {fs.root}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = handle(msg, fs)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
