"""SDP Stage 14 — SDPPipeline.

Chains every SDP component into a single ingest call:

    raw turn(s)
      → ConversationParser    (normalize)
      → SemanticExtractor     (regex facts)
      → EntityExtractor       (entities)
      → RelationshipMapper    (entities + relations)
      → ImportanceScorer      (per-unit importance)
      → ConfidenceScorer      (per-unit confidence)
      → SemanticSummarizer    (per-unit summary)
      → AALMemory list        (canonical objects)

Caller can then convert to MemoryItem with `m.to_memory_item()` and hand
them to the existing JsonlMemoryStore + Retriever (see mcp_server.py's
sdp_ingest tool).

NO LLM calls. Pure regex/heuristic. Sub-50ms typical end-to-end on a
single (user_query, assistant_reply) turn.
"""
import re
from datetime import datetime, timezone
from typing import Iterable

from orchestration.sdp.aal import AALMemory
from orchestration.sdp.extractor import SemanticExtractor
from orchestration.sdp.linker import EntityExtractor, RelationshipMapper
from orchestration.sdp.parser import ConversationParser
from orchestration.sdp.scorer import ConfidenceScorer, ImportanceScorer
from orchestration.sdp.summarizer import SemanticSummarizer


def _content_for_unit(ext: dict) -> str:
    """Build a single-sentence content line for a SemanticUnit."""
    cat, val, subj, attr = ext["category"], ext["value"], ext.get("subject"), ext.get("attribute")
    if cat == "version" and subj:
        return f"{subj} version is {val}."
    if cat == "port":
        return f"port {val} is in use."
    if cat == "url":
        return f"URL: {val}."
    if cat == "person" and attr:
        return f"{val} ({attr})."
    if cat == "technology":
        return f"Uses {val}."
    if cat == "name":
        return f"The user's name is {val}."
    if cat == "role":
        return f"The user's role is {val}."
    if cat == "employer":
        return f"The user works at {val}."
    if cat == "location":
        return f"The user is located in {val}."
    if cat == "email":
        return f"The user's email is {val}."
    if cat == "statement":
        # Sentence-level claim fallback: the sentence IS the claim. Keeping it
        # verbatim preserves rare tokens (license keys, codenames, numbers)
        # for the FTS leg, which the skeletal one-liners above lose.
        return val
    if cat == "retraction":
        return f"No longer uses {val}."
    return f"{cat}: {val}"


# ---------------------------------------------------------------------------
# Sentence-level claim fallback
#
# The categorical extractors above are precise but skeletal: a dense sentence
# like "The API gateway timeout is set to 30 seconds" produces NOTHING (no
# tech term, no port, no person), and "the importance floor is 0.4" collapses
# into "version: 0.4". Any factual sentence whose signal tokens (numbers,
# identifiers, proper nouns) are not already captured by a categorical
# extraction is stored verbatim as a "statement" claim.
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# A verb that marks a declarative factual sentence.
_FACT_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|runs?|ran|running|uses?|used|owns?|owned|"
    r"manages?|managed|equals?|costs?|takes?|targets?|set to|stays?|keeps?|kept|"
    r"stores?|stored|escalates?|supersed\w+|deploys?|deployed|hosts?|hosted|"
    r"lives?|moved|migrat\w+|switch\w+|renamed|decided|expires?|rotates?|"
    r"target(?:s|ing|ed)?|aim(?:s|ing|ed)?|plans?|planning)\b",
    re.IGNORECASE,
)
# Signal tokens: things worth remembering exactly.
_NUM_RE = re.compile(r"\b\d+(?:[.:]\d+)*%?\b")
_IDENT_RE = re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)[A-Za-z][A-Za-z0-9_\-]{2,}\b")
# Hyphenated technical terms ("belief-revision", "write-gate") carry meaning
# even when lowercase.
_HYPHEN_RE = re.compile(r"\b[a-z]\w*(?:-\w+)+\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
_QUESTION_RE = re.compile(r"\?\s*$")
# Explicit supersession cues — tier-1 of the conflict spec. Only these mark a
# claim as superseding earlier beliefs; similarity alone never does.
_SUPERSEDE_CUE_RE = re.compile(
    r"\b(no longer|not anymore|moved (?:off|away from)|moved to|migrat(?:ed|ing)|"
    r"switch(?:ed|ing)?\s+(?:to|from)|instead of|chang(?:ed|ing)\s+(?:to|from)|"
    r"renamed|deprecated|replaced (?:by|with)|now (?:runs?|uses?|lives?|on|at)|"
    r"used to (?:be|run|use|live))\b",
    re.IGNORECASE,
)
# "moved off Heroku" / "no longer uses Redis" must NOT produce a positive
# "Uses Heroku." claim — the mention is departure context, not an assertion.
def _negated_mention(span: str, value: str) -> bool:
    # Deliberately NO bare "not"/"never": "Heroku is not slow" must not
    # retract Heroku. Only explicit departure/replacement cues qualify —
    # a missed retraction leaves a stale belief (recoverable, surfaced as a
    # conflict); a false retraction destroys a true one (silent data loss).
    pat = (
        r"(?:no longer|not anymore|moved (?:off|away from)|"
        r"migrat\w+ (?:from|off)|switch\w+ (?:from|away from)|instead of|"
        r"replaced|deprecat\w+|dropp\w+|"
        r"used to (?:be|run|use|live on))\W+(?:\w+\W+){0,3}?" + re.escape(value)
    )
    return re.search(pat, span or "", re.IGNORECASE) is not None


_STOPWORD_PROPERS = {
    "The", "This", "That", "These", "Those", "It", "We", "Our", "They", "There",
    "He", "She", "You", "Your", "My", "But", "And", "Also", "However", "When",
    "What", "Where", "Who", "How", "Why", "If", "For", "With", "After", "Before",
}


def _signal_tokens(text: str) -> set[str]:
    """Tokens that carry exact factual weight: numbers, identifiers,
    proper nouns (minus sentence-grammar stopwords)."""
    out: set[str] = set()
    out.update(m.group(0).lower() for m in _NUM_RE.finditer(text))
    out.update(m.group(0).lower() for m in _IDENT_RE.finditer(text))
    out.update(m.group(0).lower() for m in _HYPHEN_RE.finditer(text))
    out.update(
        m.group(0).lower()
        for m in _PROPER_RE.finditer(text)
        if m.group(0) not in _STOPWORD_PROPERS
    )
    return out


def extract_statement_units(text: str, covered_extractions: list[dict]) -> list[dict]:
    """Sentence-level fallback claims for factual sentences whose signal
    tokens the categorical extractors did not capture.

    Returns extraction-shaped dicts (category="statement"). Each carries
    ``supersedes_hint`` when the sentence contains an explicit supersession
    cue ("no longer", "moved off", "switched to", ...).
    """
    covered = set()
    for ext in covered_extractions:
        for field in ("value", "subject", "attribute"):
            v = ext.get(field)
            if v:
                covered.update(_signal_tokens(str(v)))
                covered.add(str(v).lower())

    out: list[dict] = []
    for sent in _SENT_SPLIT_RE.split(text):
        sent = sent.strip()
        if len(sent) < 25 or len(sent) > 400:
            continue
        if _QUESTION_RE.search(sent):
            continue
        if not _FACT_VERB_RE.search(sent):
            continue
        signals = _signal_tokens(sent)
        if not signals:
            continue
        residual = signals - covered
        # Coverage only suppresses SHORT sentences. A long factual sentence
        # carries relations between its tokens ("the importance floor is 0.4")
        # that a skeletal categorical extraction ("version: 0.4") lost — keep
        # the sentence even when its individual tokens look covered.
        if not residual and len(sent) < 60:
            continue
        out.append({
            "type": "fact",
            "category": "statement",
            "value": sent,
            "subject": None,
            "attribute": None,
            "span": sent,
            "supersedes_hint": bool(_SUPERSEDE_CUE_RE.search(sent)),
        })
    return out


class SDPPipeline:
    """Lightweight semantic ingestion pipeline (no LLM)."""

    def __init__(
        self,
        parser: ConversationParser | None = None,
        extractor: SemanticExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
        relationship_mapper: RelationshipMapper | None = None,
        importance_scorer: ImportanceScorer | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        summarizer: SemanticSummarizer | None = None,
        source_tag: str = "sdp",
    ) -> None:
        self.parser = parser or ConversationParser()
        self.extractor = extractor or SemanticExtractor()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relationship_mapper = relationship_mapper or RelationshipMapper()
        self.importance_scorer = importance_scorer or ImportanceScorer()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.summarizer = summarizer or SemanticSummarizer()
        self.source_tag = source_tag

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def process_turn(
        self, user_query: str, assistant_reply: str,
        timestamp: datetime | None = None,
    ) -> list[AALMemory]:
        """Run the full SDP pipeline on one (user, assistant) exchange.

        Returns a list of AALMemory objects, one per extracted SemanticUnit.
        Empty list when no durable facts were found (e.g. pure pleasantry).
        """
        return self.process_messages([
            {"role": "user", "content": user_query, "timestamp": timestamp},
            {"role": "assistant", "content": assistant_reply, "timestamp": timestamp},
        ])

    def process_messages(self, messages: Iterable[dict]) -> list[AALMemory]:
        """Run SDP on an arbitrary list of conversation messages."""
        parsed = self.parser.parse(list(messages))
        if not parsed:
            return []

        # We extract over the whole-turn text so the linker can see both
        # halves of the exchange when finding relationships.
        joined = " ".join(m["content"] for m in parsed)
        ts = self._best_timestamp(parsed)

        # Stage 6: entities + Stage 10: relationships (computed once per turn)
        entities = self.entity_extractor.extract(joined)
        relationships = self.relationship_mapper.map(joined, entities)

        # Stage 2: per-clause extractions. A value mentioned only as departure
        # context ("moved off Heroku") must not produce a positive claim —
        # it becomes a RETRACTION unit instead, which the write gate uses to
        # close ALL active claims about that value (not just the nearest one).
        extractions = []
        for e in self.extractor.extract(joined):
            val = e.get("value")
            if val and _negated_mention(e.get("span") or "", str(val)):
                extractions.append({
                    "type": "retraction",
                    "category": "retraction",
                    "value": str(val),
                    "subject": e.get("subject"),
                    "attribute": None,
                    "span": e.get("span"),
                    "supersedes_hint": True,
                })
            else:
                extractions.append(e)

        # Stage 2b: sentence-level claim fallback — factual sentences whose
        # signal tokens (numbers, identifiers, proper nouns) the categorical
        # extractors missed are kept verbatim as "statement" claims. This is
        # what preserves dense multi-fact turns and rare tokens. Runs per
        # message (not on `joined`) so sentence-splitting never glues the end
        # of the user query onto the start of the assistant reply.
        for m in parsed:
            extractions = extractions + extract_statement_units(
                m["content"], extractions
            )
        if not extractions:
            return []

        out: list[AALMemory] = []
        for ext in extractions:
            # Compose a one-sentence content line for the SemanticUnit.
            content = _content_for_unit(ext)
            # Score importance + confidence (Stages 7 + 8).
            importance = self.importance_scorer.score(ext)
            confidence = self.confidence_scorer.score(ext)
            # Summarize the source span (Stage 9). When the content itself
            # is already a one-liner, summarize the surrounding span for
            # provenance/context.
            summary = self.summarizer.summarize(ext.get("span") or content)
            subject = ext.get("subject")
            attribute = ext.get("attribute")
            value = ext.get("value")
            if ext.get("category") == "person":
                # The person extractor puts the NAME in value and the role in
                # attribute. Re-map to a proper (entity, attribute, value)
                # claim — (Priya, role, owner) — so entity resolution links
                # "Priya" / "Priya Sharma" claims to one canonical entity.
                name = str(value or "").strip()
                if name in _STOPWORD_PROPERS:
                    continue  # "Who owns ..." is a question word, not a person
                subject, value = name, (attribute or "person")
                attribute = "role"
            out.append(AALMemory(
                content=content,
                summary=summary,
                importance=importance,
                confidence=confidence,
                entities=[e.as_dict() for e in entities],
                relationships=[r.as_dict() for r in relationships],
                type=ext["type"],
                subject=subject,
                attribute=attribute,
                value=value,
                source=self.source_tag,
                timestamp=ts,
                # Statement units carry the cue directly; categorical units
                # inherit it from their source clause — "moved off Heroku;
                # now runs on GCP" should let "Uses GCP." supersede too.
                supersedes_hint=bool(
                    ext.get("supersedes_hint")
                    or _SUPERSEDE_CUE_RE.search(ext.get("span") or "")
                ),
            ))
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _best_timestamp(parsed: list[dict]) -> datetime:
        for m in parsed:
            ts = m.get("timestamp")
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except ValueError:
                    continue
            if isinstance(ts, datetime):
                return ts
        return datetime.now(timezone.utc)
