"""SDP Stages 6 + 10 — EntityExtractor + RelationshipMapper.

Identifies entities (technologies, services, people, versions, URLs)
and produces directed relationships between them — the seed of a
semantic graph.

The doc shows spaCy NER for EntityExtractor; we use regex/keyword matching
instead so SDP has no extra dependencies. spaCy can be added later as an
optional richer extractor.
"""
import re
from dataclasses import dataclass, field

from orchestration.sdp.extractor import _TECH_RE, _PERSON_RE


_SERVICE_HINT_RE = re.compile(
    r"\b([a-z][a-z0-9_-]*[-_](?:svc|service|gateway|api|server|db|store|queue|worker|job))\b",
    re.IGNORECASE,
)
_VERSION_NEAR_TECH_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9.+]+)\s+(v?\d+\.\d+(?:\.\d+)?)\b"
)


@dataclass
class Entity:
    text: str
    type: str  # technology | service | person | version | url | other
    normalized: str | None = None

    def as_dict(self) -> dict:
        return {"text": self.text, "type": self.type, "normalized": self.normalized}


@dataclass
class Relationship:
    source: str
    relation: str
    target: str
    confidence: float = 0.7
    evidence: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source, "relation": self.relation, "target": self.target,
            "confidence": self.confidence, "evidence": self.evidence,
        }


class EntityExtractor:
    """Regex-based entity extraction across the vocab in extractor._TECH_RE."""

    def extract(self, text: str) -> list[Entity]:
        if not text:
            return []
        seen: dict[tuple[str, str], Entity] = {}

        for m in _TECH_RE.finditer(text):
            term = m.group(1)
            key = ("technology", term.lower())
            if key not in seen:
                seen[key] = Entity(text=term, type="technology", normalized=term)

        for m in _SERVICE_HINT_RE.finditer(text):
            name = m.group(1)
            key = ("service", name.lower())
            if key not in seen:
                seen[key] = Entity(text=name, type="service", normalized=name)

        for m in _PERSON_RE.finditer(text):
            name = m.group(1)
            key = ("person", name.lower())
            if key not in seen:
                seen[key] = Entity(text=name, type="person", normalized=name)

        for m in _VERSION_NEAR_TECH_RE.finditer(text):
            preceding, version = m.group(1), m.group(2)
            if preceding.lower() in {t.lower() for t in seen.keys() for t in [t[1]]}:
                # The preceding term is already known — emit a version entity
                key = ("version", version)
                if key not in seen:
                    seen[key] = Entity(text=version, type="version", normalized=version)

        return list(seen.values())


class RelationshipMapper:
    """Produce directed relationships from a list of entities + the source text.

    Heuristics:
      - "X uses Y" / "X on Y" → uses(X, Y)
      - "X runs Y" → runs(X, Y)
      - "migrated from X to Y" → replaced_by(X, Y)
      - Same span containing a service + a technology → uses(service, tech)
      - Same span containing a tech + a version → version(tech, version)
    """

    _USES_RE = re.compile(
        r"\b([A-Za-z][A-Za-z0-9.+_-]+)\s+(?:uses?|using|runs on|hosted on|deployed on|"
        r"lives on|is on|sits on)\s+([A-Za-z][A-Za-z0-9.+_-]+)\b",
        re.IGNORECASE,
    )
    _MIGRATION_RE = re.compile(
        r"\bmigrat(?:ed|ing) (?:from )?([A-Za-z][A-Za-z0-9.+_-]+)\s+to\s+([A-Za-z][A-Za-z0-9.+_-]+)\b",
        re.IGNORECASE,
    )
    _UPGRADE_RE = re.compile(
        r"\b(?:upgrade[d]?|moved)\s+([A-Za-z][A-Za-z0-9.+_-]+)\s+to\s+([A-Za-z0-9.+]+)\b",
        re.IGNORECASE,
    )

    def map(self, text: str, entities: list[Entity]) -> list[Relationship]:
        if not text:
            return []
        rels: list[Relationship] = []

        for m in self._USES_RE.finditer(text):
            src, tgt = m.group(1).strip(), m.group(2).strip()
            rels.append(Relationship(source=src, relation="uses", target=tgt,
                                     confidence=0.8, evidence=m.group(0)))

        for m in self._MIGRATION_RE.finditer(text):
            old, new = m.group(1).strip(), m.group(2).strip()
            rels.append(Relationship(source=old, relation="replaced_by", target=new,
                                     confidence=0.85, evidence=m.group(0)))

        for m in self._UPGRADE_RE.finditer(text):
            src, tgt = m.group(1).strip(), m.group(2).strip()
            rels.append(Relationship(source=src, relation="upgraded_to", target=tgt,
                                     confidence=0.85, evidence=m.group(0)))

        # Co-occurrence: any service + technology pair in the same text gets a weak uses() link
        services = [e for e in entities if e.type == "service"]
        techs = [e for e in entities if e.type == "technology"]
        for svc in services:
            for tech in techs:
                # Skip ones we already wrote via explicit verbs
                if any(r.source.lower() == svc.text.lower() and r.target.lower() == tech.text.lower()
                       for r in rels):
                    continue
                rels.append(Relationship(
                    source=svc.text, relation="uses", target=tech.text,
                    confidence=0.5, evidence="(co-occurrence)",
                ))

        # Versions: tech + nearby version becomes a version() relation
        for m in _VERSION_NEAR_TECH_RE.finditer(text):
            preceding, version = m.group(1).strip(), m.group(2).strip()
            if any(e.type == "technology" and e.text.lower() == preceding.lower() for e in entities):
                rels.append(Relationship(
                    source=preceding, relation="version", target=version,
                    confidence=0.85, evidence=m.group(0),
                ))

        # Deduplicate (source, relation, target) keeping highest-confidence
        best: dict[tuple[str, str, str], Relationship] = {}
        for r in rels:
            key = (r.source.lower(), r.relation, r.target.lower())
            if key not in best or best[key].confidence < r.confidence:
                best[key] = r
        return list(best.values())
