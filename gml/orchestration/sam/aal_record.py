"""AALRecord — SAM's compressed output format for one completed turn.

This is the canonical shape SAM produces from a (user_query, assistant_reply)
exchange. SDPWriter consumes AALRecords and writes them to the store +
retriever + entity index.

An AALRecord has TWO halves:

  * ``tuples``         — list of structured facts in (subject, verb, object,
                         time, negated) form. These embed cleanly and win
                         on precise factual recall.
  * ``chunk_summary``  — a short, casual-prose compression of the turn that
                         preserves conversational flow without pleasantries.
                         Embeds as natural-sounding sentences and wins on
                         topic / context recall.

Storing both gives the retriever two angles on the same turn — fact-shaped
embedding AND chunk-shaped embedding. Authority scores differentiate them
in the reranker (tuples 0.85, chunks 0.75).

This dataclass is the wire format between SAM and SDP. Frozen so two
producers can hand the same record to one writer without race.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable


@dataclass
class AALTuple:
    """One factual claim extracted by SAM from the turn."""

    subject: str
    verb: str
    object: str
    time: str | None = None
    negated: bool = False
    confidence: float = 0.85

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
            "time": self.time,
            "negated": self.negated,
            "confidence": self.confidence,
        }

    def to_content(self) -> str:
        parts = [self.subject]
        if self.negated:
            parts.append("did not")
        parts.append(self.verb)
        parts.append(self.object)
        base = " ".join(parts)
        if self.time:
            base = f"{base} ({self.time})"
        return base


@dataclass
class AALRecord:
    """SAM's per-turn output, ready for SDPWriter to persist."""

    # Structured facts
    tuples: list[AALTuple] = field(default_factory=list)

    # Conversational chunk
    chunk_summary: str = ""
    chunk_user: str = ""
    chunk_assistant: str = ""

    # Metadata
    entities: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | int | None = None
    dia_id_user: str | None = None
    dia_id_assistant: str | None = None

    # Scoring
    importance: float = 0.7
    confidence: float = 0.8

    @property
    def is_empty(self) -> bool:
        """True when there's nothing worth persisting."""
        return not self.tuples and not self.chunk_summary.strip()

    def as_dict(self) -> dict:
        return {
            "tuples": [t.as_dict() for t in self.tuples],
            "chunk_summary": self.chunk_summary,
            "chunk_user": self.chunk_user,
            "chunk_assistant": self.chunk_assistant,
            "entities": list(self.entities),
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "dia_id_user": self.dia_id_user,
            "dia_id_assistant": self.dia_id_assistant,
            "importance": self.importance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AALRecord":
        """Inverse of as_dict — useful for tests, replay, RPC."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now(timezone.utc)
        return cls(
            tuples=[
                AALTuple(
                    subject=t["subject"],
                    verb=t["verb"],
                    object=t["object"],
                    time=t.get("time"),
                    negated=bool(t.get("negated", False)),
                    confidence=float(t.get("confidence", 0.85)),
                )
                for t in data.get("tuples", [])
            ],
            chunk_summary=data.get("chunk_summary", ""),
            chunk_user=data.get("chunk_user", ""),
            chunk_assistant=data.get("chunk_assistant", ""),
            entities=list(data.get("entities", [])),
            timestamp=ts if isinstance(ts, datetime) else datetime.now(timezone.utc),
            session_id=data.get("session_id"),
            dia_id_user=data.get("dia_id_user"),
            dia_id_assistant=data.get("dia_id_assistant"),
            importance=float(data.get("importance", 0.7)),
            confidence=float(data.get("confidence", 0.8)),
        )
