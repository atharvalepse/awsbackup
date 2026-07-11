"""Memory extractor — turns a (user_query, assistant_reply) turn into
:class:`MemoryItem` records using a local LLM (default: DeepSeek R1 8B via
the same Ollama client SAM uses).

The extractor prompts the model for structured entity/attribute/value
extractions. Each extraction becomes a MemoryItem; the Conversation runner
hands them to the :class:`MemoryStore`.

When the LLM is unavailable or the output can't be parsed, the extractor
returns an empty list — never raises into the Conversation hot path.
"""
import json
import os
import re
import uuid
from datetime import datetime, timezone

from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import MemoryItem
from orchestration.sam._ollama_client import OllamaClient


slog = StructuredLogger("ingestion.extractor")

# Assistant-introduced facts are softer evidence than user-stated ones, so
# their LLM confidence is scaled by this factor (they still rank below an
# equally-confident user fact). Overridable via env.
_ASSISTANT_CONFIDENCE_FACTOR = float(
    os.environ.get("GML_ASSISTANT_CONFIDENCE_FACTOR", "0.7")
)


DEFAULT_AUTHORITY = 0.6  # used only when the LLM doesn't return a per-fact confidence


# The prompt deliberately uses ABSTRACT placeholders ("<a tool name>") for its
# illustrative examples — not concrete strings. An earlier version listed
# "we deployed v2.4.1 to staging yesterday" verbatim and DeepSeek-R1 then
# regurgitated that exact sentence into real users' memory stores as if it
# were their own fact (the duplicate-v2.4.1 bug). Treat any concrete example
# in this prompt as a leak risk.
_EXTRACTION_PROMPT = """\
You are a memory extractor for a long-term AI memory system. You are shown
exactly ONE user→assistant conversation turn. Your job: extract user-facing
factual claims worth recalling in a future conversation.

Output JSON ONLY, no markdown fences, no commentary. Schema:

{{
  "memories": [
    {{
      "content":       "<one full-sentence claim, written in third person>",
      "entity":        "<the subject of the claim, or null>",
      "attribute":     "<which property of the entity, or null>",
      "value":         "<the value of that property, or null>",
      "confidence":    <float in [0.0, 1.0] — see scale below>,
      "speaker":       "user" | "assistant" — who asserted this fact,
      "summary_short": "<6-12 word version, or null>"
    }}
  ]
}}

ATOMICITY (most-violated rule -- read carefully):
A memory holds ONE fact. If the user turn states multiple independent
facts joined by "and", "also", commas, or separate clauses, emit ONE
memory per fact. Examples (abstract):
  "I work at <org>, my lead is <person>, and we use <tool>"
    -->  THREE memories, not one.
  "<entity-A> ships on <day>; <entity-B> ships on <other-day>"
    -->  TWO memories, not one.
Never collapse multiple facts into one "content" string with conjunctions.

SPEAKER ATTRIBUTION (label every fact — frequently wrong):
Extract durable factual claims from BOTH sides and set "speaker":
* "user"      — the user stated it about themselves / their world.
* "assistant" — the assistant introduced it (e.g. answered the user's
                question with a concrete fact).
  user: "what's the deploy target again?"
  asst: "we usually deploy to <region>"
    -->  ONE memory, speaker="assistant", value=<region>.
Be CONSERVATIVE with assistant facts: extract only concrete, durable,
specific claims worth recalling later (configs, identifiers, decisions,
data, definitions). Do NOT extract the assistant's opinions, hedges,
generic advice, disclaimers, or its restatement of the user's question.
If the user and the assistant state the SAME fact, extract it ONCE with
speaker="user".

DO extract (when stated by the USER):
* User preferences / habits / decisions — INCLUDING negative ones.
  "I don't drink coffee", "we never deploy on Friday", "I hate dark mode"
  are all real user facts, NOT denials. Extract them.
* Concrete facts about systems, tools, projects, people, schedules,
  locations, identifiers, configurations, versions.
* Events the user mentioned ("we shipped <X> last <day>").
* Identity claims the user made about themselves.

DO NOT extract:
* Anything the user merely ASKED ("what's the deploy target?" is a
  question, not a fact -- emit nothing).
* Assistant disclaimers or admissions of ignorance.
  e.g. "I don't have your name on file", "I'm not sure", "as an AI...".
  (These are about what the AI doesn't know -- distinct from the user
  stating they don't do something, which IS a fact about the user.)
* Pure pleasantries / acknowledgements -- "Hi", "Thanks", "Got it",
  "Noted", "Sure", "Sounds good", "You're welcome".
* Hypothetical / speculative content the user hasn't committed to --
  "what if we used X", "we could maybe switch to Y".
* Time-of-day chatter ("good morning", weather), unless the user
  bound it to a fact ("standup is at 10am").

CONFIDENCE scale (USE THE FULL RANGE -- do not default to 0.95):
  0.95-1.00  user stated it as a definite fact, no hedging.
             ("My name is <X>", "we ship Fridays")
  0.80-0.94  user stated it but with mild hedge or in passing.
             ("I think we use <X>", "btw my editor is <Y>")
  0.65-0.79  user implied it without saying outright.
             ("yeah staging-east-3 most of the time idk" -- hedged)
  0.40-0.64  inferred from context, plausible but not certain.
  <0.40      do NOT include it -- too speculative.
If the user uses hedging words ("idk", "I think", "maybe", "kinda",
"basically", "btw"), drop confidence into the 0.70-0.85 band.

VERBATIM TERMINOLOGY (read this -- it is the most common hallucination):
Copy the user's terms EXACTLY as written. Do NOT:
  * autocorrect typos          ("nvim" stays "nvim", not "vim")
  * normalize abbreviations    ("k8s" stays "k8s", not "Kubernetes")
  * expand acronyms            ("MCP" stays "MCP", not "Model Context...")
  * substitute similar words   ("/memories" stays "/memories", not "/memory")
  * change casing               ("FastAPI" stays "FastAPI", not "fastapi")
Tools, file paths, version strings, person names, region codes,
URLs, command names -- ALL must appear in your output exactly as in
user_query. If unsure of the exact word, COPY IT CHARACTER-BY-CHARACTER
from user_query rather than rewording.

TEMPORAL PHRASES — keep them, don't skip the memory because of them.
The user's exact time word ("yesterday", "last Friday", "this week") is
the correct value to emit. Do not omit a memory just because it mentions
time, and do not convert relative phrases to absolute calendar dates.

Worked examples (study these — each is a real failure mode you must avoid):

  user: "my fav editor is nvim btw"
    GOOD: [{{"content": "user's favorite editor is nvim", ...}}]
    BAD:  "user's favorite editor is vim"                  (lost the n)
    BAD:  []                                                (over-cautious)

  user: "the /memories endpoint broke yesterday"
    GOOD: [{{"content": "the /memories endpoint broke yesterday", ...}}]
    BAD:  "the /memories endpoint broke on 2023-10-15"     (invented date)
    BAD:  []                                                (don't skip the memory)

  user: "we deploy via k8s"
    GOOD: [{{"content": "they deploy via k8s", ...}}]
    BAD:  "they deploy via Kubernetes"                     (expanded acronym)

  user: "I work at TronoCity Labs and my boss is Bharat and we use Postgres"
    GOOD: THREE separate memories -- one per fact.
    BAD:  one memory combining all three with "and".

User query (verbatim):
{user_query!r}

Assistant reply (verbatim):
{assistant_reply!r}

If the turn yields zero memories that pass the rules above, return
{{"memories": []}}.
"""


def _build_prompt(user_query: str, assistant_reply: str) -> str:
    return _EXTRACTION_PROMPT.format(
        user_query=user_query, assistant_reply=assistant_reply
    )


def _parse_extraction(answer: str) -> list[dict]:
    text = answer.strip()
    if not text:
        return []
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    open_idx = text.find("{")
    close_idx = text.rfind("}")
    if open_idx < 0 or close_idx <= open_idx:
        return []
    try:
        data = json.loads(text[open_idx : close_idx + 1])
    except json.JSONDecodeError:
        return []
    raw = data.get("memories", [])
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict) and m.get("content")]


# Belt-and-braces filter that catches obvious assistant-denial / pleasantry
# output the LLM might emit despite the prompt. Cheap and deterministic.
_DENIAL_PATTERNS = (
    "i don't have", "i do not have", "i don't know", "i do not know",
    "i'm not sure", "i am not sure", "as an ai", "as an llm",
    "i can't help", "i cannot help", "i can't access", "i cannot access",
    "i'm unable to", "i am unable to", "i'm sorry, but",
    "got it",  # the assistant's acknowledgement, not a fact
    "noted", "sounds good", "you're welcome", "happy to help",
)


def _looks_like_denial_or_pleasantry(content: str) -> bool:
    lc = content.strip().lower()
    if not lc:
        return True
    # Very short utterances are almost certainly pleasantries.
    if len(lc) <= 12 and any(p in lc for p in ("hi", "hello", "thanks", "thank you")):
        return True
    return any(p in lc for p in _DENIAL_PATTERNS)


# Question-detection guard. The model frequently misattributes the assistant's
# answer to the user when the user_query is itself a question — e.g. user asks
# "what's the deploy target?" and the assistant says "staging-east-3", and the
# model then emits "the deploy target is staging-east-3" as if the USER said
# it. The prompt rule alone doesn't hold; this Python filter is the backstop.
_WH_PREFIXES = (
    "what", "when", "where", "how", "why", "who", "which", "whose",
    "is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ",
    "should ", "will ",
)
_USER_CLAIM_MARKERS = (
    "i ", "i'm", "im ", "my ", "we ", "we're", "were ", "our ",
    "i've", "ive ", "i'll", "ill ", "i'd ",
)


# Hallucinated-date guard. The model frequently converts relative time words
# ("yesterday", "last friday") into invented absolute dates like "2023-10-15".
# When that happens and the user's actual phrasing was relative, restore the
# user's exact phrase. When the user said nothing temporal at all, strip the
# absolute date entirely (don't replace -- we have no source of truth for it).
_ABS_DATE_RE = re.compile(
    r"\b(?:on\s+)?"
    r"(?:\d{4}-\d{1,2}-\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?"
    r"|\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.IGNORECASE,
)
_RELATIVE_TIME_PHRASES = (
    "yesterday", "today", "tomorrow", "tonight", "this morning",
    "this afternoon", "this evening", "this week", "last week",
    "next week", "this month", "last month", "next month",
    "earlier", "just now", "a moment ago", "recently",
    "last monday", "last tuesday", "last wednesday", "last thursday",
    "last friday", "last saturday", "last sunday",
    "this monday", "this tuesday", "this wednesday", "this thursday",
    "this friday", "this saturday", "this sunday",
)


def _strip_hallucinated_dates(content: str, user_query: str) -> str:
    """Replace LLM-invented absolute dates with the user's relative phrase
    (or strip them if the user didn't reference time at all)."""
    if not _ABS_DATE_RE.search(content):
        return content
    uq_lc = user_query.lower()
    user_phrase = next(
        (p for p in _RELATIVE_TIME_PHRASES if p in uq_lc),
        None,
    )
    if user_phrase is None:
        # User said nothing temporal; the model made up a date.
        # Strip it (with any leading "on " connector) and tidy whitespace.
        return re.sub(r"\s+", " ", _ABS_DATE_RE.sub("", content)).strip(" ,.;")
    # User used a relative phrase; put it back where the date was.
    return _ABS_DATE_RE.sub(user_phrase, content)


def _user_query_is_pure_question(user_query: str) -> bool:
    """True if user_query reads as a question with no first-person claim.

    The intuition: if the user asked something and made no claim of their own,
    nothing in this turn is a user-stated fact -- any extraction must be
    coming from the assistant's reply, which we don't want.
    """
    s = user_query.strip().lower()
    if not s:
        return False
    # Does it have any user-claim marker? If yes, it's not a pure question
    # ("I work at X but what's the deploy target?" still claims something).
    if any(marker in (" " + s + " ") for marker in _USER_CLAIM_MARKERS):
        return False
    # Ends with ? → almost certainly a question.
    if s.rstrip(" .!").endswith("?"):
        return True
    # Starts with a WH-word / aux verb → also a question even without "?".
    return any(s.startswith(p) for p in _WH_PREFIXES)


def _coerce_confidence(raw, fallback: float) -> float:
    """Clamp the LLM-returned confidence to a sane float in [0,1]."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class MemoryExtractor:
    """LLM-driven extractor producing MemoryItems from a single turn.

    ``client`` is the same :class:`OllamaClient` SAM uses (default
    ``deepseek-r1:8b``). Pass any other implementation for tests.

    Example:
        >>> extractor = MemoryExtractor(client=HTTPOllamaClient())
        >>> items = await extractor.extract(
        ...     user_query="we ship Friday",
        ...     assistant_reply="Got it; freezing main from Thursday EOD.",
        ...     session_id="s-1",
        ... )
    """

    def __init__(
        self,
        client: OllamaClient,
        authority_score: float = DEFAULT_AUTHORITY,
        source: str = "conversation",
    ) -> None:
        self.client = client
        self.authority_score = authority_score
        self.source = source

    async def extract(
        self,
        user_query: str,
        assistant_reply: str,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        prompt = _build_prompt(user_query, assistant_reply)
        try:
            gen = await self.client.generate(prompt, json_mode=True, temperature=0.0)
        except Exception as exc:
            slog.warning(
                event="extractor_llm_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                degraded_mode=True,
            )
            return []

        extractions = _parse_extraction(gen.answer)
        # Speaker-attribution backstop: if the user_query is a pure question
        # with no first-person claim, any extracted fact must have come from
        # the assistant's reply, so force speaker="assistant" (the LLM often
        # mislabels these as user facts). We now KEEP assistant facts —
        # tagged source="assistant" and confidence-discounted — rather than
        # dropping the batch.
        force_assistant = bool(extractions) and _user_query_is_pure_question(user_query)
        now = datetime.now(timezone.utc)
        items: list[MemoryItem] = []
        dropped_denial = 0
        dropped_dates = 0
        for ext in extractions:
            content = (ext.get("content") or "").strip()
            if not content:
                continue
            if _looks_like_denial_or_pleasantry(content):
                dropped_denial += 1
                continue
            # Restore relative time phrases the model converted to absolute
            # dates (or strip dates if the user said nothing temporal).
            fixed = _strip_hallucinated_dates(content, user_query)
            if fixed != content:
                dropped_dates += 1
                content = fixed
            # Per-fact confidence comes from the LLM. The constructor's
            # `authority_score` is only a fallback for the no-confidence
            # case (and for tests that don't set it).
            authority = _coerce_confidence(
                ext.get("confidence"), fallback=self.authority_score,
            )
            # Who asserted it. Pure-question turns force "assistant" (backstop);
            # otherwise trust the LLM's per-fact label, defaulting to "user".
            speaker = "assistant" if force_assistant else (
                ext.get("speaker") or "user"
            ).strip().lower()
            is_assistant = speaker == "assistant"
            # Assistant-introduced facts are softer evidence than user-stated
            # ones — discount confidence and tag the source so retrieval/UI can
            # tell them apart.
            if is_assistant:
                authority = round(authority * _ASSISTANT_CONFIDENCE_FACTOR, 4)
            source = "assistant" if is_assistant else self.source
            try:
                items.append(MemoryItem(
                    id=f"conv-{uuid.uuid4().hex[:12]}",
                    content=content,
                    summary_short=(ext.get("summary_short") or None),
                    entity=(ext.get("entity") or None),
                    attribute=(ext.get("attribute") or None),
                    value=(ext.get("value") or None),
                    timestamp=now,
                    source=source,
                    authority_score=authority,
                    pinned=False,
                    raw_metadata={
                        "session_id": session_id,
                        "extracted_by": "deepseek-r1:8b",
                        "speaker": speaker,
                        # Keep the LLM's raw value too, for audit / future
                        # rerank tuning. authority_score is the canonical one.
                        "confidence": authority,
                    },
                ))
            except Exception as exc:
                slog.warning(
                    event="extractor_record_invalid_skipping",
                    error=str(exc),
                    raw=ext,
                )
                continue
        if dropped_denial:
            slog.info(event="extractor_dropped_denials", count=dropped_denial)
        if dropped_dates:
            slog.info(event="extractor_repaired_dates", count=dropped_dates)
        return items
