-- 010_is_latest.sql — soft supersession flag for memories.
--
-- Retrieval filters on is_latest = TRUE so superseded facts don't surface on
-- the fast /memory/recall path (which, unlike the full pipeline, does NOT run
-- SAM's read-time conflict resolution).
--
-- Defaults TRUE so this is additive and backward-compatible: every existing
-- row stays retrievable, and nothing is hidden until a writer explicitly marks
-- a row superseded. Auto-supersession on ingest is OPT-IN
-- (GML_SUPERSEDE_ON_INGEST) — attribute granularity is coarse (many distinct
-- facts share e.g. entity='user', attribute='preference'), so eager
-- supersession would wrongly hide valid memories.

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS is_latest BOOLEAN NOT NULL DEFAULT TRUE;

-- Retrieval almost always filters is_latest = TRUE; a partial index keeps that
-- cheap. Use CREATE INDEX CONCURRENTLY when applying to a populated prod table
-- (cannot run inside a txn) to avoid an ACCESS EXCLUSIVE lock.
CREATE INDEX IF NOT EXISTS idx_memories_is_latest
    ON memories (user_id) WHERE is_latest;
