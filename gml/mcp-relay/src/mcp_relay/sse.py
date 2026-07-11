"""Server-Sent Events helpers shared by the relay, the connector, and tests."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional


def format_sse(data: Any, event: Optional[str] = "message", id: Optional[str] = None) -> bytes:
    """Encode a single SSE event. `data` is JSON-serialised."""
    lines = []
    if id is not None:
        lines.append(f"id: {id}")
    if event is not None:
        lines.append(f"event: {event}")
    payload = json.dumps(data, separators=(",", ":"))
    # A data value may not contain raw newlines; split into multiple data: lines.
    for chunk in payload.split("\n"):
        lines.append(f"data: {chunk}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def format_comment(text: str = "") -> bytes:
    """A keepalive comment line; ignored by SSE parsers but keeps the socket warm."""
    return f": {text}\n\n".encode("utf-8")


async def iter_sse(line_iter: AsyncIterator[str]) -> AsyncIterator[tuple[str, Any]]:
    """Parse an async iterator of lines (e.g. httpx ``response.aiter_lines()``) into
    ``(event, data)`` tuples, where ``data`` is JSON-decoded. Comments are skipped."""
    event = "message"
    data_lines: list[str] = []
    async for raw in line_iter:
        line = raw.rstrip("\r")
        if line == "":
            # Dispatch on blank line if we accumulated any data.
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    data = raw_data
                yield event, data
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue  # comment / keepalive
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
        # other fields (id:, retry:) are ignored for our purposes
