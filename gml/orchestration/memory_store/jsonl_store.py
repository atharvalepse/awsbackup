"""JSON-Lines :class:`MemoryStore` — one record per line, append-only.

Why JSONL: durable, diffable, trivially inspectable with `cat`, no DB
dependency, survives crashes mid-write (the broken line is dropped at
next load). Records carry pre-computed vectors inline when present so
the Retriever can skip re-embedding at startup.

File format: each line is a JSON object matching :class:`MemoryItem`'s
``model_dump_json()`` output. The store is single-process; concurrent
writes from multiple processes WILL interleave and produce corrupt lines.
Wrap in a higher-level lock if you need multi-process safety.

The public methods are ``async def`` to match the MemoryStore ABC, but the
underlying file I/O is synchronous and offloaded to a threadpool via
``asyncio.to_thread`` — keeps the event loop free under concurrent ingests.
"""
import asyncio
import json
from pathlib import Path

from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import MemoryItem
from orchestration.memory_store.base import MemoryStore


slog = StructuredLogger("memory_store.jsonl")


class JsonlMemoryStore(MemoryStore):
    """Append-only JSONL store keyed by ``MemoryItem.id``. Single-tenant —
    ignores the ``user_id`` parameter on async methods (kept on the
    signature so the interface matches the Postgres adapter)."""

    def __init__(self, path: str | Path, create_if_missing: bool = True) -> None:
        self.path = Path(path)
        if create_if_missing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    # ----------------------------------------------------------------------
    # Async API (the only API now — sync helpers are private, prefixed _)
    # ----------------------------------------------------------------------

    async def load_all(self, user_id: str | None = None) -> list[MemoryItem]:
        return await asyncio.to_thread(self._load_all_sync)

    async def add(self, item: MemoryItem, user_id: str | None = None) -> None:
        await asyncio.to_thread(self._add_sync, item)

    async def add_many(
        self, items: list[MemoryItem], user_id: str | None = None
    ) -> None:
        if not items:
            return
        # Split over-long content into sentence-aligned chunks sharing a
        # parent_memory_id (no-op for normal atomic facts).
        from orchestration.ingestion.chunking import expand_chunked
        await asyncio.to_thread(self._add_many_sync, expand_chunked(items))

    async def delete(self, memory_id: str, user_id: str | None = None) -> bool:
        return await asyncio.to_thread(self._delete_sync, memory_id)

    # ----------------------------------------------------------------------
    # Sync internals — never call these from async code directly. Tests can.
    # ----------------------------------------------------------------------

    def _load_all_sync(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        records: list[MemoryItem] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    records.append(MemoryItem.model_validate(data))
                except (json.JSONDecodeError, ValueError) as exc:
                    slog.warning(
                        event="jsonl_record_invalid_skipping",
                        path=str(self.path),
                        line=line_no,
                        error=str(exc),
                        degraded_mode=True,
                    )
                    continue
        return records

    def _add_sync(self, item: MemoryItem) -> None:
        line = item.model_dump_json()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _add_many_sync(self, items: list[MemoryItem]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

    def _delete_sync(self, memory_id: str) -> bool:
        """Rewrites the file atomically (temp file + ``os.replace``) so a
        crash mid-write can't truncate the store — readers either see the
        old file or the new one, never a half-written one."""
        records = self._load_all_sync()
        remaining = [r for r in records if r.id != memory_id]
        if len(remaining) == len(records):
            return False
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in remaining:
                f.write(r.model_dump_json() + "\n")
        tmp.replace(self.path)  # atomic on POSIX; replaces existing file
        slog.info(event="jsonl_record_deleted", path=str(self.path), memory_id=memory_id)
        return True
