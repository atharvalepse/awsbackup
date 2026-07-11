"""SAMTurnCompressor — SAM compresses one (user, assistant) turn into AAL.

This is the producer side of the SAM → AAL → SDP pipeline. Given a
completed turn, SAM emits an :class:`AALRecord` containing:

  * structured ``tuples`` (factual claims for entity-indexed lookup)
  * a casual-prose ``chunk_summary`` (for dense / topic retrieval)
  * extracted ``entities`` (people, dates, places)
  * an ``importance`` and ``confidence`` score

One LLM call per turn, JSON-mode output, with a tolerant parser that
falls back to an empty record on any error (so ingest never blocks).

Used by the bench's ``sam-aal`` ingest mode and by the live MCP
``ingest()`` path going forward. Replaces the standalone
``AALTupleExtractor`` as the canonical AAL producer.
"""
import json
import re
import uuid
from datetime import datetime, timezone

from orchestration.observability.logging import StructuredLogger
from orchestration.sam._ollama_client import OllamaClient
from orchestration.sam.aal_record import AALRecord, AALTuple


slog = StructuredLogger("sam.turn_compressor")


_SESSION_SUMMARY_PROMPT = """\
You summarize one CONVERSATION SESSION into a single short sentence
suitable for topic-level memory retrieval.

A session is a series of messages between two speakers. Read all of them.
Then produce ONE sentence (≤ 30 words) that captures what the session
was ABOUT — the central topic, the people involved, and any concrete
durable facts mentioned. Avoid pleasantries. Third person. Casual.

Session messages (in order):
{session_text}

Respond with JSON ONLY:
{{"summary": "...", "topic": "<3-5 word topic>", "entities": ["...", "..."], "importance": 0.8}}

If the session is purely pleasantries with no content, return:
{{"summary": "", "topic": "", "entities": [], "importance": 0.1}}
"""


_COMPRESS_PROMPT = """\
You compress one conversation turn into a structured JSON record. The
record will be stored as long-term memory for an AI assistant.

A turn is a USER message followed by an ASSISTANT reply.

Your job is to emit:
  1. ``tuples`` — every concrete factual claim as a (subject, verb,
     object, time, negated) record. Use NORMALIZED base-form verbs
     ('research' not 'looking into', 'go' not 'went', 'use' not 'using').
     Skip pleasantries and pure feelings.
  2. ``chunk_summary`` — ONE short sentence (≤ 25 words) summarizing
     the salient content of the turn, in third person, no pleasantries.
     Should preserve who-did-what-when in casual natural language.
  3. ``entities`` — list of all named entities mentioned (people, places,
     dates, products, services), lowercased.
  4. ``importance`` (0..1) — how likely this turn matters in future
     conversations. Pure greetings = 0.2, durable facts = 0.9.
  5. ``confidence`` (0..1) — how sure you are the extraction is correct.

EXAMPLES:

USER: "Hey Mel! Been looking into adoption agencies all weekend."
ASSISTANT: "Wow Caroline, that's a big step. Researching specific ones?"
→ {{"tuples": [{{"subject": "Caroline", "verb": "research", "object": "adoption agencies", "time": "weekend", "negated": false}}], "chunk_summary": "Caroline researched adoption agencies over the weekend.", "entities": ["caroline", "mel", "adoption agencies"], "importance": 0.85, "confidence": 0.9}}

USER: "I went to the LGBTQ support group on Sunday, May 7."
ASSISTANT: "How did it go?"
→ {{"tuples": [{{"subject": "Caroline", "verb": "attend", "object": "LGBTQ support group", "time": "Sunday May 7", "negated": false}}], "chunk_summary": "Caroline attended the LGBTQ support group on Sunday May 7.", "entities": ["caroline", "lgbtq support group", "sunday may 7"], "importance": 0.9, "confidence": 0.95}}

USER: "We don't use Redis for sessions anymore."
ASSISTANT: "Got it, what now?"
→ {{"tuples": [{{"subject": "session cache", "verb": "use", "object": "Redis", "time": null, "negated": true}}], "chunk_summary": "Session cache no longer uses Redis.", "entities": ["redis", "session cache"], "importance": 0.85, "confidence": 0.9}}

USER: "Hi! How are you?"
ASSISTANT: "Hey, doing great, you?"
→ {{"tuples": [], "chunk_summary": "", "entities": [], "importance": 0.1, "confidence": 0.5}}

NOW DO THE SAME FOR THIS TURN:

USER: {user_text!r}

ASSISTANT: {assistant_text!r}

Respond with JSON ONLY. No prose, no markdown. Exact shape:

{{
  "tuples": [
    {{"subject": "...", "verb": "...", "object": "...", "time": "..." or null, "negated": false}}
  ],
  "chunk_summary": "...",
  "entities": ["...", "..."],
  "importance": 0.8,
  "confidence": 0.9
}}
"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text


def _parse_aal_response(answer: str) -> dict | None:
    """Tolerant JSON extractor — find outermost braces, parse, validate."""
    if not answer:
        return None
    text = _strip_fence(answer)
    o = text.find("{")
    c = text.rfind("}")
    if o < 0 or c <= o:
        return None
    try:
        return json.loads(text[o : c + 1])
    except json.JSONDecodeError:
        return None


def _normalize_tuple_dict(raw: dict) -> AALTuple | None:
    """Validate one tuple dict and convert into AALTuple. Returns None on bad input."""
    if not isinstance(raw, dict):
        return None
    subj = (raw.get("subject") or "").strip()
    verb = (raw.get("verb") or "").strip().lower()
    obj = (raw.get("object") or "").strip()
    if not subj or not verb or not obj:
        return None
    if subj.lower() in {"speaker a", "speaker b", "user", "assistant", "someone", "anyone"}:
        return None
    return AALTuple(
        subject=subj, verb=verb, object=obj,
        time=(raw.get("time") or None),
        negated=bool(raw.get("negated", False)),
        confidence=float(raw.get("confidence", 0.85)),
    )


class SAMTurnCompressor:
    """One LLM call → AALRecord. SAM as the compressor.

    Args:
        client: any :class:`OllamaClient` (Ollama or llama.cpp).
        default_authority: the importance baseline if the LLM doesn't supply one.
    """

    def __init__(
        self,
        client: OllamaClient,
        default_authority: float = 0.7,
    ) -> None:
        self.client = client
        self.default_authority = default_authority

    async def compress(
        self,
        user_text: str,
        assistant_text: str,
        *,
        timestamp: datetime | None = None,
        session_id: str | int | None = None,
        dia_id_user: str | None = None,
        dia_id_assistant: str | None = None,
    ) -> AALRecord:
        """Compress one turn into an :class:`AALRecord`. Never raises."""
        ts = timestamp or datetime.now(timezone.utc)

        if not user_text and not assistant_text:
            return AALRecord(timestamp=ts, session_id=session_id)

        prompt = _COMPRESS_PROMPT.format(
            user_text=user_text or "",
            assistant_text=assistant_text or "",
        )

        try:
            gen = await self.client.generate(prompt, json_mode=True)
        except Exception as exc:
            slog.warning(
                event="turn_compressor_llm_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                degraded_mode=True,
            )
            return AALRecord(
                timestamp=ts, session_id=session_id,
                dia_id_user=dia_id_user, dia_id_assistant=dia_id_assistant,
                chunk_user=user_text, chunk_assistant=assistant_text,
            )

        data = _parse_aal_response(gen.answer)
        if not isinstance(data, dict):
            slog.warning(event="turn_compressor_unparseable", raw=(gen.answer or "")[:200])
            return AALRecord(
                timestamp=ts, session_id=session_id,
                dia_id_user=dia_id_user, dia_id_assistant=dia_id_assistant,
                chunk_user=user_text, chunk_assistant=assistant_text,
            )

        # Tuples
        raw_tuples = data.get("tuples", []) or []
        tuples: list[AALTuple] = []
        for raw in raw_tuples if isinstance(raw_tuples, list) else []:
            t = _normalize_tuple_dict(raw)
            if t is not None:
                tuples.append(t)

        # Chunk summary — strip, cap length
        chunk = (data.get("chunk_summary") or "").strip()
        if len(chunk) > 400:
            chunk = chunk[:399] + "…"

        # Entities — list of lowercased strings
        entities_raw = data.get("entities", []) or []
        entities: list[str] = []
        if isinstance(entities_raw, list):
            for e in entities_raw:
                if isinstance(e, str) and e.strip():
                    entities.append(e.strip().lower())
        # Dedup preserving order
        seen = set()
        entities = [e for e in entities if not (e in seen or seen.add(e))]

        try:
            importance = float(data.get("importance", self.default_authority))
            importance = max(0.0, min(1.0, importance))
        except (TypeError, ValueError):
            importance = self.default_authority
        try:
            confidence = float(data.get("confidence", 0.85))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.85

        return AALRecord(
            tuples=tuples,
            chunk_summary=chunk,
            chunk_user=user_text,
            chunk_assistant=assistant_text,
            entities=entities,
            timestamp=ts,
            session_id=session_id,
            dia_id_user=dia_id_user,
            dia_id_assistant=dia_id_assistant,
            importance=importance,
            confidence=confidence,
        )

    async def summarize_session(
        self,
        messages: list[dict],
        *,
        timestamp: datetime | None = None,
        session_id: str | int | None = None,
    ) -> AALRecord:
        """Phase #2: compress an entire session into a topic-level AAL record.

        Useful for "what was the conversation about" / open-domain questions
        (LOCOMO cat-4). The session summary embeds as one dense memory at
        topic granularity. Returns an AALRecord with `chunk_summary` set and
        no tuples (the per-turn compress calls handle those).
        """
        ts = timestamp or datetime.now(timezone.utc)
        if not messages:
            return AALRecord(timestamp=ts, session_id=session_id)

        # Render the session compactly; cap to ~40 messages or ~3500 chars
        # to fit Qwen's context window comfortably.
        lines: list[str] = []
        total_chars = 0
        for m in messages[:40]:
            speaker = (m.get("speaker") or "S").strip()
            text = (m.get("content") or m.get("text") or "").strip()
            if not text:
                continue
            line = f"{speaker}: {text}"
            if total_chars + len(line) > 3500:
                break
            lines.append(line)
            total_chars += len(line) + 1
        session_text = "\n".join(lines)
        if not session_text:
            return AALRecord(timestamp=ts, session_id=session_id)

        prompt = _SESSION_SUMMARY_PROMPT.format(session_text=session_text)
        try:
            gen = await self.client.generate(prompt, json_mode=True)
        except Exception as exc:
            slog.warning(
                event="session_summary_llm_failed",
                error_type=type(exc).__name__, degraded_mode=True,
            )
            return AALRecord(timestamp=ts, session_id=session_id)

        data = _parse_aal_response(gen.answer)
        if not isinstance(data, dict):
            return AALRecord(timestamp=ts, session_id=session_id)

        summary = (data.get("summary") or "").strip()
        topic = (data.get("topic") or "").strip()
        if not summary:
            return AALRecord(timestamp=ts, session_id=session_id)
        if len(summary) > 400:
            summary = summary[:399] + "…"

        entities_raw = data.get("entities", []) or []
        entities: list[str] = []
        if isinstance(entities_raw, list):
            for e in entities_raw:
                if isinstance(e, str) and e.strip():
                    entities.append(e.strip().lower())
        seen = set()
        entities = [e for e in entities if not (e in seen or seen.add(e))]

        try:
            importance = float(data.get("importance", 0.80))
            importance = max(0.0, min(1.0, importance))
        except (TypeError, ValueError):
            importance = 0.80

        # Prepend the topic so embedding sees the topic word(s) explicitly.
        full_summary = f"[{topic}] {summary}" if topic else summary
        return AALRecord(
            tuples=[],
            chunk_summary=full_summary,
            chunk_user="",
            chunk_assistant="",
            entities=entities,
            timestamp=ts,
            session_id=session_id,
            importance=importance,
            confidence=0.85,
        )
