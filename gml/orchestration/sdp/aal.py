"""SDP Stages 5 + 11 — SemanticUnit + AALMemory.

`SemanticUnit` is an atomic semantic atom — the "do not store paragraphs"
rule from the doc. Each unit is one fact / decision / issue / preference.

`AALMemory` is the canonical structured-memory object produced by SDP.
It bundles a unit's content + a compressed summary + scores + entities +
relationships. The `to_memory_item()` method converts it into the
existing project's `MemoryItem` (in pipeline/contracts.py) so SDP plugs
straight into the existing store + retriever without a new schema.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orchestration.pipeline.contracts import MemoryItem


@dataclass
class SemanticUnit:
    """An atomic semantic claim — never a paragraph."""

    type: str                          # fact | decision | issue | preference | config
    value: str                         # the actual claim, ideally one sentence
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"type": self.type, "value": self.value, "metadata": self.metadata}


@dataclass
class AALMemory:
    """Adaptive AAL memory — canonical structured-memory object.

    Fields mirror the doc plus enough extras to round-trip into MemoryItem
    without information loss.
    """

    content: str
    summary: str
    importance: float
    confidence: float
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    type: str = "fact"
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    source: str = "sdp"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # True when the source sentence carried an explicit supersession cue
    # ("no longer", "moved off", "switched to"). The write gate only closes
    # old beliefs on this signal — similarity alone never supersedes.
    supersedes_hint: bool = False

    def as_dict(self) -> dict:
        return {
            "content": self.content, "summary": self.summary,
            "importance": self.importance, "confidence": self.confidence,
            "entities": self.entities, "relationships": self.relationships,
            "type": self.type, "subject": self.subject, "attribute": self.attribute,
            "value": self.value, "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_memory_item(self) -> MemoryItem:
        """Convert into the project's existing MemoryItem contract.

        Mapping:
          AALMemory.content        → MemoryItem.content
          AALMemory.summary        → MemoryItem.summary_short
          AALMemory.subject        → MemoryItem.entity
          AALMemory.attribute      → MemoryItem.attribute
          AALMemory.value          → MemoryItem.value
          AALMemory.importance     → MemoryItem.authority_score
          AALMemory.confidence     → raw_metadata["confidence"]
          AALMemory.entities       → raw_metadata["entities"]
          AALMemory.relationships  → raw_metadata["relationships"]
          AALMemory.type           → raw_metadata["sdp_type"]
        """
        return MemoryItem(
            id=f"sdp-{uuid.uuid4().hex[:12]}",
            content=self.content,
            summary_short=self.summary,
            entity=self.subject,
            attribute=self.attribute,
            value=self.value,
            timestamp=self.timestamp,
            source=self.source,
            authority_score=self.importance,
            pinned=False,
            raw_metadata={
                "confidence": self.confidence,
                "sdp_type": self.type,
                "entities": self.entities,
                "relationships": self.relationships,
                **({"supersedes_hint": True} if self.supersedes_hint else {}),
            },
        )
