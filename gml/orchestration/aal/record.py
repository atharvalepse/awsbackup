"""AAL — the canonical persisted memory format.

An AAL record carries TWO synchronized views of the same fact:

  * ``simplemem``  one-line natural sentence — what a human would say to
                   re-create the fact. Embeds well; survives prose-flavoured
                   retrieval queries.

  * ``sjson``      structured JSON triple — explicit subject/verb/object
                   with a small fixed schema. Wins on precise factual
                   lookup ("what version of X did we pick?") and lets the
                   reranker / SAM reason over discrete entities rather than
                   parsing prose.

The two are NEVER independent. Both are produced (or filled with safe
defaults) for every persisted memory. The retrieval and rerank stacks can
look at either side at no cost — they live on the same row.

Wire-format
-----------
The dataclass serializes 1:1 onto a :class:`MemoryItem` row:

    simplemem  →  MemoryItem.content
    sjson      →  MemoryItem.{entity, attribute, value} + raw_metadata["sjson"]
                  (raw_metadata also carries time/negated/confidence/extra)

Postgres
--------
Migration 008 adds dedicated ``aal_simplemem`` + ``aal_sjson`` columns so
the canonical form is queryable + indexable without parsing raw_metadata.
JSONL fallback uses the same MemoryItem schema and stores AAL in the
``aal_simplemem`` + ``aal_sjson`` MemoryItem fields directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from orchestration.pipeline.contracts import MemoryItem


# Minimum sjson schema. Producers should fill what they know and leave the
# rest at the default. Extra keys (e.g. "category", "evidence_span") are
# allowed and survive through the store via the extras pass-through.
_SJSON_REQUIRED_KEYS: tuple[str, ...] = (
    "subject",   # who/what the fact is about (entity)
    "verb",      # the relation — verbs are normalized in the LLM extractor
    "object",    # the value / target
)
_SJSON_OPTIONAL_KEYS: tuple[str, ...] = (
    "time",           # ISO date or natural phrase (e.g. "last quarter")
    "negated",        # True if the fact is a negation ("we DON'T use X")
    "confidence",     # [0, 1] — how sure the extractor is
    "category",       # SDP-side category tag (e.g. "version", "port", "url")
    "evidence_span",  # the original turn text the fact came from
)


@dataclass
class AAL:
    """Atomic memory record — one fact in two synchronized views."""

    simplemem: str
    sjson: dict[str, Any] = field(default_factory=dict)

    # Provenance / metadata that lives alongside the AAL — kept separate
    # from `sjson` so the structured triple stays clean.
    importance: float = 0.7
    source: str = "conversation"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str | None = None  # if None, the writer assigns a uuid

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Make sjson always a dict, never None — simplifies downstream code.
        if self.sjson is None:
            self.sjson = {}
        # Clamp confidence into the [0, 1] band if present.
        conf = self.sjson.get("confidence")
        if conf is not None:
            self.sjson["confidence"] = max(0.0, min(1.0, float(conf)))
        # Negated coerces from various LLM outputs ("yes" / "true" / 1 / True)
        # to a real bool so the downstream reranker can rely on it.
        if "negated" in self.sjson:
            self.sjson["negated"] = _coerce_bool(self.sjson["negated"])

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def is_well_formed(self) -> bool:
        """True iff at least ``subject`` AND ``object`` are present in sjson.

        A triple with no subject and no object is a free-form sentence —
        still storable as simplemem-only, but the structured side adds
        nothing. Callers can use this to decide whether to pay for the
        structured-index cost.
        """
        return bool(self.sjson.get("subject") and self.sjson.get("object"))

    # ------------------------------------------------------------------
    # Conversion to/from MemoryItem (the persisted row shape)
    # ------------------------------------------------------------------

    def to_memory_item(self) -> MemoryItem:
        """Map AAL onto a MemoryItem row.

        - simplemem → MemoryItem.content
        - sjson["subject"]   → MemoryItem.entity
        - sjson["verb"]      → MemoryItem.attribute
        - sjson["object"]    → MemoryItem.value
        - The full sjson (plus a marker tag) is also pickled into
          raw_metadata so a future reader can reconstruct the AAL
          losslessly even from a row written by an older code path.
        """
        import uuid as _uuid

        mid = self.id or f"aal-{_uuid.uuid4().hex[:12]}"
        sjson = dict(self.sjson)  # defensive copy
        raw_meta: dict[str, Any] = {
            "format": "aal",
            "sjson": sjson,
        }
        # If the writer attached extra metadata, preserve it under a nested key
        # rather than risking collision with reserved fields.
        if "extra" in sjson:
            raw_meta["extra"] = sjson.pop("extra")

        return MemoryItem(
            id=mid,
            content=self.simplemem,
            entity=_str_or_none(sjson.get("subject")),
            attribute=_str_or_none(sjson.get("verb")),
            value=_str_or_none(sjson.get("object")),
            source=self.source,
            authority_score=max(0.0, min(1.0, float(self.importance))),
            pinned=False,
            timestamp=self.timestamp,
            raw_metadata=raw_meta,
            # Canonical AAL columns — Postgres writes these to dedicated
            # columns (migration 008). JSONL stores them alongside the rest.
            aal_simplemem=self.simplemem,
            aal_sjson=dict(sjson),
        )

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "AAL":
        """Reconstruct an AAL from a stored MemoryItem.

        Works on rows written by this code (raw_metadata['format'] == 'aal')
        AND on legacy rows that pre-date AAL (we synthesize sjson from
        entity/attribute/value).
        """
        # Preferred source: the dedicated AAL columns. Fall back to
        # raw_metadata['sjson'] and finally to the {entity, attribute,
        # value} triple for legacy rows.
        if item.aal_simplemem is not None or item.aal_sjson is not None:
            return cls(
                simplemem=item.aal_simplemem or item.content,
                sjson=dict(item.aal_sjson or {}),
                importance=float(item.authority_score),
                source=item.source,
                timestamp=item.timestamp,
                id=item.id,
            )
        rm = item.raw_metadata or {}
        sjson = dict(rm.get("sjson") or {})
        if not sjson:
            if item.entity:
                sjson["subject"] = item.entity
            if item.attribute:
                sjson["verb"] = item.attribute
            if item.value:
                sjson["object"] = item.value
        return cls(
            simplemem=item.content,
            sjson=sjson,
            importance=float(item.authority_score),
            source=item.source,
            timestamp=item.timestamp,
            id=item.id,
        )

    # ------------------------------------------------------------------
    # Serialization helpers (for the API + bench logs)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = {
            "simplemem": self.simplemem,
            "sjson": dict(self.sjson),
            "importance": self.importance,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.id:
            d["id"] = self.id
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ----------------------------------------------------------------------
# AAL bundle — a turn's worth of AAL records
# ----------------------------------------------------------------------


@dataclass
class AALBundle:
    """Set of AAL records extracted from a single (user, assistant) turn.

    A turn typically produces 0..N AALs depending on how factual it was.
    Carrying them as a bundle lets the ingest endpoint commit them as one
    transaction and report a single summary line.
    """

    records: list[AAL] = field(default_factory=list)
    turn_id: str | None = None
    trace_id: str | None = None

    def to_memory_items(self) -> list[MemoryItem]:
        return [r.to_memory_item() for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _coerce_bool(value: Any) -> bool:
    """Tolerate LLM-output forms of bool ('yes', 'true', 1)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y", "t"}
    return False
