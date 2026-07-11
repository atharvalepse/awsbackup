"""Per-user API key store — JSONL-backed, no-DB design.

Why this exists: ``GML_API_KEY`` was a single shared secret that worked for
"is this caller authorized at all?" but had no notion of WHICH user is calling.
With this module the server can:

  * accept N per-user keys
  * map each key to a stable ``user_id`` (used downstream for memory namespacing
    when that lands)
  * issue/revoke keys via an admin endpoint guarded by the master key

Storage shape — one JSON object per line in ``users.jsonl``::

    {"key": "k_...", "user_id": "alice", "created_at": "2026-05-24T12:00:00Z", "label": "alice's laptop"}

A path lookup ``$GML_USER_KEYS_FILE`` overrides; otherwise we default to
``~/.gml/users.jsonl``. The file is created lazily on first write.

Concurrency: a single process-local ``threading.RLock`` guards reads and
writes. Multiple uvicorn workers would race here — for that case you want to
migrate to SQLite (cheap, single-file, FTS5 + WAL handles concurrent writes).
The interface is intentionally narrow so the migration is a one-file swap.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_USERS_PATH = Path.home() / ".gml" / "users.jsonl"


class UserRecord:
    __slots__ = ("key", "user_id", "created_at", "label")

    def __init__(
        self,
        key: str,
        user_id: str,
        created_at: str,
        label: str | None = None,
    ) -> None:
        self.key = key
        self.user_id = user_id
        self.created_at = created_at
        self.label = label

    def to_dict(self) -> dict:
        d = {"key": self.key, "user_id": self.user_id, "created_at": self.created_at}
        if self.label:
            d["label"] = self.label
        return d

    def to_public_dict(self) -> dict:
        """Same as ``to_dict`` but with the key REDACTED — safe for listing.

        Returns the key prefix + suffix so the user can identify which key is
        which without the secret leaking.
        """
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "label": self.label,
            "key_preview": f"{self.key[:8]}…{self.key[-4:]}" if len(self.key) > 14 else "…",
        }


def _resolve_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("GML_USER_KEYS_FILE", "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_USERS_PATH


class UserKeyStore:
    """Per-user API key store. Process-local lock; one file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = _resolve_path(path)
        self._lock = threading.RLock()
        self._cache: dict[str, UserRecord] | None = None

    # ----------------------------------------------------------------------

    def _load(self) -> dict[str, UserRecord]:
        if self._cache is not None:
            return self._cache
        records: dict[str, UserRecord] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "key" not in d or "user_id" not in d:
                        continue
                    rec = UserRecord(
                        key=str(d["key"]),
                        user_id=str(d["user_id"]),
                        created_at=str(d.get("created_at", "")),
                        label=d.get("label"),
                    )
                    records[rec.key] = rec
        self._cache = records
        return records

    def _persist(self) -> None:
        assert self._cache is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in self._cache.values():
                f.write(json.dumps(rec.to_dict()) + "\n")
        tmp.replace(self.path)

    # ----------------------------------------------------------------------
    # Public API — async to match PostgresUserKeyStore so callers can swap
    # backends via the `orchestration.storage` factory without touching
    # call sites. The underlying I/O is sync + cached; we await
    # `asyncio.to_thread` only when the cache misses.

    async def lookup(self, key: str) -> UserRecord | None:
        return self._lookup_sync(key)

    async def issue(self, user_id: str, label: str | None = None) -> UserRecord:
        import asyncio
        return await asyncio.to_thread(self._issue_sync, user_id, label)

    async def revoke(self, key: str) -> bool:
        import asyncio
        return await asyncio.to_thread(self._revoke_sync, key)

    async def list_users(self) -> list[UserRecord]:
        with self._lock:
            return list(self._load().values())

    async def by_user_id(self, user_id: str):
        """All keys belonging to a given user_id (async generator)."""
        with self._lock:
            for rec in self._load().values():
                if rec.user_id == user_id:
                    yield rec

    # ----------------------------------------------------------------------
    # Sync internals (used by the async wrappers above; called directly
    # by tests and one-shot scripts).

    def _lookup_sync(self, key: str) -> UserRecord | None:
        if not key:
            return None
        with self._lock:
            return self._load().get(key)

    def _issue_sync(self, user_id: str, label: str | None = None) -> UserRecord:
        if not user_id:
            raise ValueError("user_id is required")
        key = "gml_" + secrets.token_urlsafe(32)
        rec = UserRecord(
            key=key,
            user_id=user_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            label=label,
        )
        with self._lock:
            cache = self._load()
            cache[key] = rec
            self._persist()
        return rec

    def _revoke_sync(self, key: str) -> bool:
        with self._lock:
            cache = self._load()
            if key not in cache:
                return False
            del cache[key]
            self._persist()
            return True


# Module-level singleton — lazily created on first access. Tests can override
# by assigning to ``_INSTANCE`` directly.
_INSTANCE: UserKeyStore | None = None


def get_user_store() -> UserKeyStore:
    """Return the process-wide UserKeyStore (lazy)."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = UserKeyStore()
    return _INSTANCE
