"""Structured (JSON-line) logger for pipeline stages.

Emits one JSON object per log call to stdout, plus forwards to the stdlib
``logging`` module so ``caplog``-based tests and downstream handlers continue
to work. No external dependencies.

Output format example::

    {"timestamp": "2026-05-21T18:30:45.123Z", "level": "info",
     "component": "pipeline", "event": "stage_complete",
     "trace_id": "abc123", "stage": "retriever", "duration_ms": 1234}

Field order is stable: ``timestamp``, ``level``, ``component``, ``event``,
``trace_id`` (omitted entirely when ``None``), then kwargs in insertion order.
"""
import json
import logging as _stdlogging
import sys
import threading
from datetime import datetime, timezone
from typing import Any


_LOCK = threading.Lock()

_LEVEL_TO_STDLIB = {
    "debug": _stdlogging.DEBUG,
    "info": _stdlogging.INFO,
    "warning": _stdlogging.WARNING,
    "error": _stdlogging.ERROR,
}

_stdlogging.getLogger("gml").addHandler(_stdlogging.NullHandler())


# MCP servers run JSON-RPC over stdout — log lines on stdout corrupt the
# channel. `mcp_server.run()` calls `set_output_stream(sys.stderr)` before
# starting; the CLI/HTTP paths keep the default (sys.stdout, looked up at
# write time so pytest's capsys/capfd capture correctly).
_OUTPUT_STREAM = None  # None → use sys.stdout looked up each write


def set_output_stream(stream) -> None:
    """Redirect all StructuredLogger output to ``stream``. Process-global.

    Pass ``None`` to revert to the default of resolving ``sys.stdout``
    at each write — useful for tests that swap stdout via capsys.
    """
    global _OUTPUT_STREAM
    _OUTPUT_STREAM = stream


def _utc_iso_ms() -> str:
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class StructuredLogger:
    """JSON-line logger keyed by component name."""

    def __init__(self, component: str) -> None:
        self.component = component
        self._stdlib = _stdlogging.getLogger(f"gml.{component}")

    def log(
        self,
        level: str,
        event: str,
        trace_id: str | None = None,
        **data: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": _utc_iso_ms(),
            "level": level,
            "component": self.component,
            "event": event,
        }
        if trace_id is not None:
            payload["trace_id"] = trace_id
        for k, v in data.items():
            payload[k] = v

        line = json.dumps(payload, sort_keys=False, default=str)
        out = _OUTPUT_STREAM if _OUTPUT_STREAM is not None else sys.stdout
        with _LOCK:
            out.write(line + "\n")
            out.flush()

        py_level = _LEVEL_TO_STDLIB.get(level, _stdlogging.INFO)
        self._stdlib.log(py_level, line)

    def info(self, event: str, trace_id: str | None = None, **data: Any) -> None:
        self.log("info", event, trace_id, **data)

    def warning(self, event: str, trace_id: str | None = None, **data: Any) -> None:
        self.log("warning", event, trace_id, **data)

    def error(self, event: str, trace_id: str | None = None, **data: Any) -> None:
        self.log("error", event, trace_id, **data)

    def debug(self, event: str, trace_id: str | None = None, **data: Any) -> None:
        self.log("debug", event, trace_id, **data)
