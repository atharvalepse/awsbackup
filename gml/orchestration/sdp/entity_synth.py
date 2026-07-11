"""Heuristic entity synthesis — aggregate per-entity memories at index time.

Why this exists: a multi-hop question like "What hobbies does Sam have?"
needs 3-5 separate memories in top-K to recover all the items in the
gold answer. Real retrieval often surfaces only the most-prominent one
(see the trace on conv-30 where the model answered just "painting" when
gold was "painting, kayaking, hiking, cooking, running").

Fix: after all sessions for a conversation are ingested, scan the
memories, group by top-N entities, and emit ONE "About <entity>:" synth
memory per entity containing short snippets from every memory mentioning
that entity. Single retrieval hit → multi-item answer surfaces in
top-K.

No LLM at index time. Pure string matching against the entity_index
top entities. ~5-15ms per conversation.

The synth records get:
  source="aal-entity-synth"
  authority_score=0.78 (between session-summary 0.80 and chunk 0.75)
  entity=<entity name>

Usage:
    from orchestration.sdp.entity_synth import synthesize_entity_memories
    synths = synthesize_entity_memories(
        memories=all_ingested,
        top_entities=entity_index.top_entities(n=20),
    )
    for s in synths:
        await store.add(s)
        await retriever.ingest([s])
"""
import hashlib
from datetime import timezone

from orchestration.pipeline.contracts import MemoryItem


# Authority slot: between session-summary (0.80) and chunk (0.75). Lower
# than tuples (0.85) because synth is heuristic, not LLM-extracted.
_AUTHORITY = 0.78
DEFAULT_MAX_SNIPPETS = 12
DEFAULT_SNIPPET_CHARS = 120
DEFAULT_MIN_MEMORIES = 3


def synthesize_entity_memories(
    memories: list[MemoryItem],
    top_entities: list,
    *,
    max_entities: int = 8,
    max_snippets: int = DEFAULT_MAX_SNIPPETS,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    min_memories: int = DEFAULT_MIN_MEMORIES,
) -> list[MemoryItem]:
    """Emit one synth memory per top entity.

    ``top_entities`` is either:
      - list[str]               (just entity names)
      - list[tuple[str, int]]   (entity_index.top_entities() return shape)

    Skips entities that appear in fewer than ``min_memories`` distinct
    memories — a single mention isn't worth synthesizing.
    """
    if not memories or not top_entities:
        return []

    # Normalize top_entities to list[str]
    ent_names: list[str] = []
    for item in top_entities[:max_entities]:
        if isinstance(item, tuple) and item:
            ent_names.append(str(item[0]))
        elif isinstance(item, str):
            ent_names.append(item)

    synth_out: list[MemoryItem] = []
    for ent in ent_names:
        ent_norm = ent.strip()
        if not ent_norm or len(ent_norm) < 2:
            continue
        ent_lower = ent_norm.lower()

        # Find memories containing this entity (case-insensitive substring).
        # We use simple string match because the entity_index doesn't link
        # back to MemoryItem ids — it tracks counts only.
        matching: list[MemoryItem] = []
        for m in memories:
            content = (m.content or "")
            if ent_lower in content.lower():
                matching.append(m)

        if len(matching) < min_memories:
            continue

        # Sort newest-first for recency-biased snippets.
        matching.sort(key=lambda m: m.timestamp, reverse=True)

        snippets: list[str] = []
        seen_keys: set = set()
        for m in matching:
            text = (m.content or "").strip()
            if not text:
                continue
            snip = text[:snippet_chars]
            # Cheap dedupe — collapse near-duplicates (same first 40 chars)
            key = snip[:40].lower().strip()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            snippets.append(f"- {snip}")
            if len(snippets) >= max_snippets:
                break

        if len(snippets) < min_memories:
            continue

        # ID is stable across runs given the same entity name — keeps
        # idempotent ingest from spawning duplicate synth memories on
        # re-ingestion of the same conversation.
        eid_hash = hashlib.md5(f"entity-synth|{ent_lower}".encode()).hexdigest()[:12]
        content = f"About {ent_norm}:\n" + "\n".join(snippets)

        # Use the newest mention's timestamp; falls back gracefully if it's
        # naive (no tzinfo).
        ts = max(m.timestamp for m in matching)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        synth_out.append(MemoryItem(
            id=f"synth-{eid_hash}",
            content=content,
            timestamp=ts,
            source="aal-entity-synth",
            authority_score=_AUTHORITY,
            pinned=False,
            entity=ent_norm,
        ))

    return synth_out
