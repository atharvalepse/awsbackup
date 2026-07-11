"""Conversation runner — the end-to-end orchestrator.

Wires the full lifecycle of a single turn:

    user text
      → Pipeline.run                 (produces TranslatedPayload)
      → Client.send                  (target AI returns AssistantResponse)
      → MemoryExtractor.extract      (optional, LLM-driven)
      → MemoryStore.add_many         (persist to disk)
      → Retriever.ingest             (optional, if Retriever supports it
                                      and the new records have vectors)

Holds a :class:`Session` so multi-turn conversations carry session_id and
a turn history across calls.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable

from orchestration.clients.base import AssistantResponse, Client
from orchestration.ingestion.extractor import MemoryExtractor
from orchestration.memory_store.base import MemoryStore
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import MemoryItem, TargetDescriptor, TranslatedPayload
from orchestration.pipeline.pipeline import Pipeline


slog = StructuredLogger("runner")


@dataclass
class Turn:
    """A single user→assistant exchange within a Session."""

    user_query: str
    assistant_reply: str
    target: TargetDescriptor
    trace_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload_user_query: str | None = None  # SAM-rewritten query if different from user_query


@dataclass
class Session:
    """Conversation state carried across turns."""

    target: TargetDescriptor
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turns: list[Turn] = field(default_factory=list)


@dataclass
class TurnResult:
    """Everything one ``ask`` call produced — for introspection and logging."""

    response: AssistantResponse
    payload: TranslatedPayload
    extracted_memories: list[MemoryItem]


# Hook to ingest new records into a Retriever after persistence.
# Signature: async def ingest(items: list[MemoryItem]) -> None
RetrieverIngestHook = Callable[[list[MemoryItem]], Awaitable[None]]


class Conversation:
    """End-to-end driver for one Session.

    ``extractor`` and ``memory_store`` are optional — pass None to disable
    memory growth. ``retriever_ingest`` is an optional hook to add newly
    extracted memories into a live Retriever (e.g. SemanticRetriever.ingest).

    Example:
        >>> conv = Conversation(
        ...     pipeline=pipeline,
        ...     client=OllamaClient(model="deepseek-r1:8b"),
        ...     target=TargetDescriptor.for_deepseek(),
        ...     extractor=MemoryExtractor(client=HTTPOllamaClient()),
        ...     memory_store=JsonlMemoryStore("~/.gml/memories.jsonl"),
        ... )
        >>> result = await conv.ask("what was that incident about?")
        >>> print(result.response.text)
    """

    def __init__(
        self,
        pipeline: Pipeline,
        client: Client,
        target: TargetDescriptor,
        session_id: str | None = None,
        extractor: MemoryExtractor | None = None,
        memory_store: MemoryStore | None = None,
        retriever_ingest: RetrieverIngestHook | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.client = client
        self.session = Session(target=target, session_id=session_id or uuid.uuid4().hex)
        self.extractor = extractor
        self.memory_store = memory_store
        self.retriever_ingest = retriever_ingest

    async def ask(self, text: str, user_id: str | None = None) -> TurnResult:
        trace_id = uuid.uuid4().hex
        slog.info(
            event="turn_start",
            session_id=self.session.session_id,
            trace_id=trace_id,
            target_family=self.session.target.model_family.value,
        )

        query = self.pipeline.build_query(
            text=text,
            target=self.session.target,
            user_id=user_id,
            session_context={
                "session_id": self.session.session_id,
                "history": [
                    {"user": t.user_query, "assistant": t.assistant_reply}
                    for t in self.session.turns[-5:]  # last 5 turns
                ],
            },
            trace_id=trace_id,
        )

        # 1. Pipeline
        payload = await self.pipeline.run(query)

        # 2. Client → target AI
        response = await self.client.send(payload)

        # 3. Memory extraction (best-effort, never raises into caller)
        extracted: list[MemoryItem] = []
        if self.extractor is not None:
            try:
                extracted = await self.extractor.extract(
                    user_query=text,
                    assistant_reply=response.text,
                    session_id=self.session.session_id,
                )
            except Exception as exc:
                slog.warning(
                    event="extractor_raised_skipping",
                    trace_id=trace_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    degraded_mode=True,
                )

        # 4. Persist
        if extracted and self.memory_store is not None:
            await self.memory_store.add_many(extracted, user_id=user_id)

        # 5. Live ingest into Retriever (so the next turn can see them)
        if extracted and self.retriever_ingest is not None:
            try:
                await self.retriever_ingest(extracted)
            except Exception as exc:
                slog.warning(
                    event="retriever_ingest_failed",
                    trace_id=trace_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    degraded_mode=True,
                )

        # 6. Append to session history
        self.session.turns.append(Turn(
            user_query=text,
            assistant_reply=response.text,
            target=self.session.target,
            trace_id=trace_id,
            payload_user_query=payload.user_query if payload.user_query != text else None,
        ))

        slog.info(
            event="turn_complete",
            session_id=self.session.session_id,
            trace_id=trace_id,
            response_latency_ms=response.latency_ms,
            extracted_memories=len(extracted),
        )
        return TurnResult(response=response, payload=payload, extracted_memories=extracted)
