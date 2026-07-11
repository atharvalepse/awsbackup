"""AAL knowledge-tuple extractor — denser memory, verb-normalized.

Why tuples
----------
A typical LOCOMO message:

    "Caroline: I've been looking into adoption agencies all weekend,
     it's overwhelming"

SDP regex extraction misses this (no tech vocab). MemoryExtractor LLM
turns it into a free-form sentence ("Caroline has been researching
adoption agencies"). Neither captures the joint signal that lets
retrieval find this when asked "What did Caroline research?".

AAL emits *both* the natural sentence AND a structured tuple:

    {
      "subject": "Caroline",
      "verb": "research",          ← NORMALIZED — "looking into" → "research"
      "object": "adoption agencies",
      "time": "weekend before May 7 2023",
      "source": "D2:8"
    }

We store the tuple as a separate MemoryItem (entity=Caroline,
attribute=research, value="adoption agencies"). At query time, the
question "What did Caroline research?" embeds close to the tuple's
content ("Caroline research adoption agencies") AND the entity index
filters to Caroline-only candidates. Retrieval lands on it instantly.

This is the doc's AAL ("compressed JSON without losing context") in
production form. Each turn produces 0-3 tuples; storage is denser than
raw messages.

LLM
---
Uses the same llama.cpp / Qwen3.5-4B-Q4 client SAM uses. Thinking is
disabled so each turn costs ~500ms-2s. JSON-mode output keeps the
extractor deterministic.

Fail-safe: any extraction error returns an empty list — never blocks
ingest. Use the existing SDP/raw paths as the fallback.
"""
import json
import os
import re
import uuid
from datetime import datetime, timezone

from orchestration.pipeline.contracts import MemoryItem
from orchestration.sam._ollama_client import OllamaClient


_AAL_PROMPT = """\
Extract STRUCTURED knowledge tuples from this short conversation turn.
Each tuple captures one independent fact in the form:

  {{"subject": "<who or what the fact is about>",
    "verb":    "<normalized base-form action — e.g. 'research' not 'looking into', 'go' not 'went', 'use' not 'using'>",
    "object":  "<what the verb acts on>",
    "time":    "<when it happened — relative or absolute, or null>",
    "negated": <true if the turn states this fact is NOT the case>}}

ONLY extract concrete factual claims. SKIP:
- Pure greetings ("Hi", "How are you", "Good to see you")
- Pure feelings without facts ("I'm so tired")
- Hypothetical / conditional ("if we did X")

If the turn has nothing factual, return {{"tuples": []}}.

Speaker A: {user_text!r}
Speaker B: {assistant_text!r}

Return JSON ONLY in this exact shape, no markdown, no prose around it:

{{"tuples": [{{"subject": "...", "verb": "...", "object": "...", "time": "...", "negated": false}}]}}
"""


def _parse_tuples(answer: str) -> list[dict]:
    """Tolerant JSON extraction — find outermost braces, parse, validate."""
    if not answer:
        return []
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    o = text.find("{")
    c = text.rfind("}")
    if o < 0 or c <= o:
        return []
    try:
        data = json.loads(text[o : c + 1])
    except json.JSONDecodeError:
        return []
    raw = data.get("tuples", [])
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        subj = (t.get("subject") or "").strip()
        verb = (t.get("verb") or "").strip().lower()
        obj = (t.get("object") or "").strip()
        if not subj or not verb or not obj:
            continue
        # Skip placeholders the model sometimes emits
        if subj.lower() in {"speaker a", "speaker b", "someone", "anyone"}:
            continue
        out.append({
            "subject": subj,
            "verb": verb,
            "object": obj,
            "time": (t.get("time") or None),
            "negated": bool(t.get("negated", False)),
        })
    return out


def tuple_to_content(t: dict) -> str:
    """Render one tuple back to a natural sentence for embedding.

    Includes the structure flat — subject, verb, object — so cosine
    matches questions like "what did X verb?" or "X's Y" cleanly.
    """
    parts = [t["subject"]]
    if t["negated"]:
        parts.append("did not")
    parts.append(t["verb"])
    parts.append(t["object"])
    base = " ".join(parts)
    if t.get("time"):
        base += f" ({t['time']})"
    return base


class AALTupleExtractor:
    """LLM-driven (subject, verb, object, time, negated) extractor.

    Use ``extract_from_turn(user_text, assistant_text) → list[MemoryItem]``
    to get back ready-to-store memories. The returned items have:
      - content       — flat normalized sentence (good for embedding)
      - entity        — the subject
      - attribute     — the verb (lowercased, base-form)
      - value         — the object
      - source        — "aal-tuple"
      - raw_metadata  — full tuple dict + original turn for provenance
    """

    def __init__(
        self,
        client: OllamaClient,
        source_tag: str = "aal-tuple",
        # Higher than raw (0.70) and window (0.65): tuples are the densest
        # signal — pre-extracted, verb-normalized, deduped via subject+verb+object.
        authority_score: float = 0.85,
    ) -> None:
        self.client = client
        self.source_tag = source_tag
        self.authority_score = authority_score

    async def extract_from_turn(
        self,
        user_text: str,
        assistant_text: str,
        timestamp: datetime | None = None,
        session_id: int | str | None = None,
        dia_id_user: str | None = None,
        dia_id_assistant: str | None = None,
    ) -> list[MemoryItem]:
        ts = timestamp or datetime.now(timezone.utc)
        prompt = _AAL_PROMPT.format(user_text=user_text, assistant_text=assistant_text)
        try:
            gen = await self.client.generate(prompt, json_mode=True, temperature=0.0)
        except Exception:
            return []

        tuples = _parse_tuples(gen.answer)
        items: list[MemoryItem] = []
        for t in tuples:
            content = tuple_to_content(t)
            items.append(MemoryItem(
                id=f"aal-{uuid.uuid4().hex[:12]}",
                content=content,
                summary_short=content[:120],
                entity=t["subject"],
                attribute=t["verb"],
                value=t["object"],
                timestamp=ts,
                source=self.source_tag,
                authority_score=self.authority_score,
                pinned=False,
                raw_metadata={
                    "tuple": t,
                    "session_id": session_id,
                    "dia_id_user": dia_id_user,
                    "dia_id_assistant": dia_id_assistant,
                },
            ))
        return items


AAL_ENABLED_DEFAULT = os.environ.get("GML_AAL_TUPLES", "1") == "1"
