"""SDPWriter — canonical writer for SAM-produced AAL records.

Closes the SAM → AAL → SDP triangle:

  SAM compresses one turn → ``AALRecord``
                                  ↓
                     SDPWriter.write(record)
                                  ↓
        MemoryStore (disk JSONL)  +  Retriever index (vectors + BM25)
        +  EntityIndex (people + dates hash-lookup)

One method, ``write(record) → list[MemoryItem]``, that atomically:

  1. Materializes every ``AALTuple`` as a ``MemoryItem`` with
     ``source="aal-tuple"`` and ``authority_score=0.85``. The verb is
     the attribute, the object is the value — so the entity-attribute
     conflict detector and the existing reranker pick up structured
     supersession ("payments / provider = Stripe" → ... = "Adyen").
  2. Materializes the ``chunk_summary`` as a single ``MemoryItem`` with
     ``source="aal-chunk"`` and ``authority_score=0.75``. Embedding this
     gives dense topic-level recall for open-ended questions.
  3. Persists all items via ``store.add_many``.
  4. Updates the live retriever index via ``await retriever.ingest``.
  5. Updates the entity hash-index with the record's entities (people,
     dates) and any entities pulled from per-item content.

Failure semantics: best-effort. A store-write failure raises (the caller
should retry); a retriever or entity-index failure is logged and ignored
(memory still on disk, just not searchable until next restart).
"""
import uuid

from orchestration.memory_store.base import MemoryStore
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import MemoryItem
from orchestration.retriever.base import Retriever
from orchestration.sam.aal_record import AALRecord
from orchestration.sdp.entity_index import EntityIndex


slog = StructuredLogger("sdp.writer")


# Authority scores — see audit. Tuples are densest, summaries are
# topic-level high authority, chunks are mid-tier, raw messages (handled
# elsewhere) are lower.
AUTH_TUPLE = 0.85
AUTH_SESSION_SUMMARY = 0.80
AUTH_CHUNK = 0.75


class SDPWriter:
    """Write SAM-produced AAL records to the store + retriever + entity index."""

    def __init__(
        self,
        store: MemoryStore,
        retriever: Retriever,
        entity_index: EntityIndex | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.entity_index = entity_index

    async def write(self, record: AALRecord) -> list[MemoryItem]:
        """Persist an AAL record. Returns the MemoryItems written."""
        if record.is_empty:
            return []

        items: list[MemoryItem] = []

        # 1. Tuples — one MemoryItem per tuple
        for t in record.tuples:
            items.append(MemoryItem(
                id=f"aal-t-{uuid.uuid4().hex[:12]}",
                content=t.to_content(),
                summary_short=t.to_content()[:120],
                entity=t.subject,
                attribute=t.verb,
                value=t.object,
                timestamp=record.timestamp,
                source="aal-tuple",
                authority_score=AUTH_TUPLE,
                pinned=False,
                raw_metadata={
                    "tuple": t.as_dict(),
                    "session_id": record.session_id,
                    "dia_id_user": record.dia_id_user,
                    "dia_id_assistant": record.dia_id_assistant,
                },
            ))

        # 2. Chunk summary — one MemoryItem
        if record.chunk_summary.strip():
            items.append(MemoryItem(
                id=f"aal-c-{uuid.uuid4().hex[:12]}",
                content=record.chunk_summary,
                summary_short=record.chunk_summary[:120],
                entity=None,
                attribute=None,
                value=None,
                timestamp=record.timestamp,
                source="aal-chunk",
                authority_score=AUTH_CHUNK,
                pinned=False,
                raw_metadata={
                    "session_id": record.session_id,
                    "user_msg": record.chunk_user[:500],
                    "assistant_msg": record.chunk_assistant[:500],
                    "entities": record.entities,
                    "importance": record.importance,
                    "confidence": record.confidence,
                },
            ))

        if not items:
            return []

        # 3. Persist to disk
        await self.store.add_many(items)

        # 4. Live-index into retriever (vectors + BM25). Best-effort.
        try:
            await self.retriever.ingest(items)
        except Exception as exc:
            slog.warning(
                event="sdp_writer_retriever_ingest_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                degraded_mode=True,
            )

        # 5. Entity hash-index — add SAM-extracted entities directly so
        # the EntityIndex covers things the regex extractor might miss
        # (multi-word names, places, products SAM understood).
        if self.entity_index is not None:
            try:
                self.entity_index.add_many(items)
                for ent in record.entities:
                    # Manually merge SAM's extra entities (covers e.g. multi-
                    # word, lowercased-already names that the regex misses).
                    for item in items:
                        self.entity_index._by_entity.setdefault(ent, set()).add(item.id)
                        self.entity_index._by_id.setdefault(item.id, set()).add(ent)
            except Exception as exc:
                slog.warning(
                    event="sdp_writer_entity_index_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    degraded_mode=True,
                )

        slog.info(
            event="sdp_writer_wrote",
            tuples=sum(1 for i in items if i.source == "aal-tuple"),
            chunks=sum(1 for i in items if i.source == "aal-chunk"),
            session_id=record.session_id,
        )
        return items

    async def write_session_summary(self, record: AALRecord) -> list[MemoryItem]:
        """Phase #2: write a per-session topic-summary memory.

        Distinct from per-turn ``write()`` because session summaries:
          - Use source="session-summary" so they're separable in queries
          - Have higher authority (0.80 vs 0.75 chunks) so cat-4 ranking favors them
          - Carry no per-turn metadata (no dia_id, no chunk_user/assistant)

        Skips when ``record.chunk_summary`` is empty (session had nothing to summarize).
        """
        if not record.chunk_summary.strip():
            return []

        item = MemoryItem(
            id=f"sess-{uuid.uuid4().hex[:12]}",
            content=record.chunk_summary,
            summary_short=record.chunk_summary[:160],
            entity=None,
            attribute=None,
            value=None,
            timestamp=record.timestamp,
            source="session-summary",
            authority_score=AUTH_SESSION_SUMMARY,
            pinned=False,
            raw_metadata={
                "session_id": record.session_id,
                "entities": record.entities,
                "importance": record.importance,
                "confidence": record.confidence,
            },
        )

        await self.store.add_many([item])
        try:
            await self.retriever.ingest([item])
        except Exception as exc:
            slog.warning(
                event="sdp_writer_session_retriever_failed",
                error_type=type(exc).__name__, degraded_mode=True,
            )
        if self.entity_index is not None:
            try:
                self.entity_index.add(item)
                for ent in record.entities:
                    self.entity_index._by_entity.setdefault(ent, set()).add(item.id)
                    self.entity_index._by_id.setdefault(item.id, set()).add(ent)
            except Exception:
                pass

        slog.info(
            event="sdp_writer_session_summary",
            session_id=record.session_id,
            chars=len(record.chunk_summary),
        )
        return [item]
