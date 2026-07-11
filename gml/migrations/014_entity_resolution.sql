-- 014_entity_resolution.sql — canonical entities, aliases, merge candidates.
--
-- Fixes the "GML" / "Gigzs Multi-LLM Layer" / "the system" problem: claims
-- about the same subject stored under different entity strings. Resolution
-- happens at WRITE time (orchestration/storage/entity_resolution.py):
--   exact alias → acronym/containment → trigram fuzzy → gray zone creates a
--   PROVISIONAL entity plus a reviewable merge candidate. Never guess hard.
--
-- All three tables are tenant-scoped with the same RLS pattern as memories
-- (app.current_user_id / app.is_admin). Entity ids are DETERMINISTIC
-- (ent_<sha1(user_id, first_norm)>) so concurrent shuffled writes converge.
--
-- memories.entity_id is a SOFT reference (no FK): rows may predate their
-- entity, and backfill fills it lazily.
--
-- Downgrade: migrations/downgrades/014_entity_resolution_down.sql.

BEGIN;

CREATE TABLE IF NOT EXISTS entities (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(user_id),
    canonical_name TEXT NOT NULL,
    -- provisional = created from a gray-zone match; surfaced for review
    provisional    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS entities_user_canonical
    ON entities (user_id, lower(canonical_name));

CREATE TABLE IF NOT EXISTS entity_aliases (
    user_id    TEXT NOT NULL REFERENCES users(user_id),
    alias_norm TEXT NOT NULL,           -- normalized (lowered, trimmed) alias
    entity_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    source     TEXT NOT NULL DEFAULT 'write',  -- write|acronym|fuzzy|backfill|manual
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, alias_norm)
);
-- trigram index drives the fuzzy leg of resolution
CREATE INDEX IF NOT EXISTS entity_aliases_trgm
    ON entity_aliases USING gin (alias_norm gin_trgm_ops);

CREATE TABLE IF NOT EXISTS entity_merge_candidates (
    user_id    TEXT NOT NULL REFERENCES users(user_id),
    -- lexically ordered pair so (a,b) and (b,a) can't both exist
    entity_a   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    similarity REAL,
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending|merged|rejected
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, entity_a, entity_b),
    CHECK (entity_a < entity_b)
);

ALTER TABLE memories ADD COLUMN IF NOT EXISTS entity_id TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_entity_id
    ON memories (user_id, entity_id) WHERE entity_id IS NOT NULL;

-- RLS — same pattern as memories (005)
ALTER TABLE entities               ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_aliases         ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_candidates ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS entities_self ON entities;
CREATE POLICY entities_self ON entities
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );
DROP POLICY IF EXISTS entity_aliases_self ON entity_aliases;
CREATE POLICY entity_aliases_self ON entity_aliases
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );
DROP POLICY IF EXISTS entity_merge_candidates_self ON entity_merge_candidates;
CREATE POLICY entity_merge_candidates_self ON entity_merge_candidates
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );

GRANT SELECT, INSERT, UPDATE, DELETE ON entities, entity_aliases,
    entity_merge_candidates TO gml_app;

COMMIT;
