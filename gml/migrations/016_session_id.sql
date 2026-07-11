-- 016: first-class session_id on memories.
--
-- session_id lived only inside raw_metadata JSONB — unindexed, invisible
-- to the API schema, and "show me the memories from this conversation"
-- required a full scan with a ->> filter. Promote it to a real column,
-- backfill from raw_metadata, and index per tenant.
--
-- The write path (PostgresMemoryStore.add_many) now populates the column
-- directly; raw_metadata keeps the key too for backward compatibility.

BEGIN;

ALTER TABLE memories ADD COLUMN IF NOT EXISTS session_id TEXT;

UPDATE memories
SET session_id = raw_metadata->>'session_id'
WHERE session_id IS NULL
  AND raw_metadata ? 'session_id'
  AND raw_metadata->>'session_id' <> '';

CREATE INDEX IF NOT EXISTS idx_memories_session
    ON memories (user_id, session_id) WHERE session_id IS NOT NULL;

COMMIT;
