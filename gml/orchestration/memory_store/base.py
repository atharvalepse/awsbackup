"""Persistent memory store interface.

The store owns persistence of :class:`MemoryItem` records between pipeline
runs. The :class:`SemanticRetriever` (or any other Retriever) consumes
records via :meth:`load_all` at startup; the Conversation runner persists
new records via :meth:`add` after each turn.

The interface is **async** so both implementations (JSONL via threadpool,
Postgres via asyncpg) compose cleanly with FastAPI's async handlers and
with the async pipeline.
"""
from abc import ABC, abstractmethod

from orchestration.pipeline.contracts import MemoryItem


class MemoryStore(ABC):
    @abstractmethod
    async def load_all(self, user_id: str | None = None) -> list[MemoryItem]:
        """Return persisted records, in insertion order.

        When ``user_id`` is provided, only that user's records are returned
        (Postgres implementations enforce this via RLS; JSONL is single-tenant
        and ignores the parameter).
        """

    @abstractmethod
    async def add(self, item: MemoryItem, user_id: str | None = None) -> None:
        """Append one record and flush to durable storage."""

    @abstractmethod
    async def add_many(
        self, items: list[MemoryItem], user_id: str | None = None
    ) -> None:
        """Batch append. Implementations should flush once at the end."""

    @abstractmethod
    async def delete(self, memory_id: str, user_id: str | None = None) -> bool:
        """Remove the record by id. Returns True if a row was deleted."""
