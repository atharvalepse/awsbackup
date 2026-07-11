"""AALConverter — turns ingest input (raw text or extracted facts) into AAL.

This is the INPUT-side counterpart to ``orchestration.translator.Translator``.

  * Translator   : MemoryItem  → string for a target AI         (output side)
  * AALConverter : raw input   → AAL bundle → MemoryItem rows   (input side)

Two ingest paths feed AALConverter:

  1. ``from_extracted_items(items)`` — wraps already-extracted MemoryItem
     rows (e.g. from the LLM MemoryExtractor) into AAL. Cheap: just
     synthesizes the structured triple from the row's entity/attribute/value
     and reuses the row's content as simplemem.

  2. ``from_turn(user_query, assistant_reply, ...)`` — the path the
     ``/api/memory/sdp_ingest`` endpoint will use. Runs SDP to produce
     AALMemory-shaped extractions, then maps each to a canonical AAL.

The converter never persists. It returns an :class:`AALBundle`; the caller
hands the bundle to the MemoryStore (which writes a row per AAL).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orchestration.aal.record import AAL, AALBundle
from orchestration.pipeline.contracts import MemoryItem


class AALConverter:
    """Stateless converter — pure transformations, safe to instantiate once."""

    # ------------------------------------------------------------------
    # Path 1: wrap items the LLM extractor already produced
    # ------------------------------------------------------------------

    def from_extracted_items(
        self,
        items: list[MemoryItem],
        *,
        turn_id: str | None = None,
        trace_id: str | None = None,
    ) -> AALBundle:
        """Lift a list of MemoryItem rows into AAL.

        Use when an upstream extractor (e.g. the LLM ``MemoryExtractor``)
        produced MemoryItems directly. We synthesize the sjson triple
        from the row's existing entity/attribute/value fields.
        """
        records = [self._memory_item_to_aal(it) for it in items]
        return AALBundle(records=records, turn_id=turn_id, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Path 2: convert a raw conversational turn via SDP
    # ------------------------------------------------------------------

    def from_turn(
        self,
        user_query: str,
        assistant_reply: str,
        *,
        sdp_pipeline=None,
        turn_id: str | None = None,
        trace_id: str | None = None,
    ) -> AALBundle:
        """Run SDP over a (user, assistant) turn and lift each AALMemory
        the pipeline produces into a canonical :class:`AAL`.

        ``sdp_pipeline`` is the existing
        :class:`orchestration.sdp.SDPPipeline` instance held by the server
        state. Passing it in (rather than constructing here) keeps the
        AAL package decoupled from SDP — easier to test in isolation.
        """
        if sdp_pipeline is None:
            return AALBundle(records=[], turn_id=turn_id, trace_id=trace_id)

        aal_memories = sdp_pipeline.process_turn(user_query, assistant_reply)
        records = [self._sdp_memory_to_aal(m) for m in aal_memories]
        return AALBundle(records=records, turn_id=turn_id, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Internal mappers
    # ------------------------------------------------------------------

    def _memory_item_to_aal(self, item: MemoryItem) -> AAL:
        """Lift an already-shaped MemoryItem into an AAL.

        simplemem = item.content. sjson synthesized from the existing
        entity/attribute/value triple. confidence carried from raw_metadata
        if present (the SDP and LLM extractors both write it there).
        """
        rm = item.raw_metadata or {}
        sjson: dict[str, Any] = {}
        if item.entity:
            sjson["subject"] = item.entity
        if item.attribute:
            sjson["verb"] = item.attribute
        if item.value:
            sjson["object"] = item.value
        # Carry confidence and any LLM-emitted extras through unchanged.
        if "confidence" in rm:
            sjson["confidence"] = rm["confidence"]
        if "negated" in rm:
            sjson["negated"] = rm["negated"]
        if "category" in rm:
            sjson["category"] = rm["category"]
        return AAL(
            simplemem=item.content,
            sjson=sjson,
            importance=float(item.authority_score),
            source=item.source,
            timestamp=item.timestamp,
            id=item.id,
        )

    def _sdp_memory_to_aal(self, sdp_memory) -> AAL:
        """Map an :class:`orchestration.sdp.aal.AALMemory` (legacy name —
        we're keeping it as the SDP intermediate) onto a canonical AAL.

        This is the bridge between the SDP regex-extraction stage and the
        new persisted format. SDP's ``AALMemory`` already carries content,
        importance, confidence, and structured fields — we just rename and
        reshape into the {simplemem, sjson} dual view.
        """
        # SDP's AALMemory fields: content, summary, importance, confidence,
        # entities, relationships, type, subject, attribute, value, source,
        # timestamp.
        sjson: dict[str, Any] = {}
        if getattr(sdp_memory, "subject", None):
            sjson["subject"] = sdp_memory.subject
        if getattr(sdp_memory, "attribute", None):
            sjson["verb"] = sdp_memory.attribute
        if getattr(sdp_memory, "value", None):
            sjson["object"] = sdp_memory.value
        conf = getattr(sdp_memory, "confidence", None)
        if conf is not None:
            sjson["confidence"] = float(conf)
        unit_type = getattr(sdp_memory, "type", None)
        if unit_type:
            sjson["category"] = unit_type
        if getattr(sdp_memory, "supersedes_hint", False):
            sjson["supersedes_hint"] = True

        ts = getattr(sdp_memory, "timestamp", None) or datetime.now(timezone.utc)
        return AAL(
            simplemem=sdp_memory.content,
            sjson=sjson,
            importance=float(getattr(sdp_memory, "importance", 0.7)),
            source=getattr(sdp_memory, "source", "conversation"),
            timestamp=ts,
        )
