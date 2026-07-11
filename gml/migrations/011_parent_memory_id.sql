-- 011_parent_memory_id.sql — chunk grouping for long memories.
--
-- When a memory's content exceeds the chunk budget it is split into several
-- rows on insert, each sharing parent_memory_id = the original memory id, so
-- retrieval can deduplicate chunks from the same source.
--
-- Deliberately a SOFT reference (nullable, no FK constraint): the "parent" is
-- a grouping key, not necessarily a stored row, so a hard FK would reject the
-- chunk rows. None for normal atomic facts (this is a no-op for them).

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS parent_memory_id TEXT;

CREATE INDEX IF NOT EXISTS idx_memories_parent
    ON memories (user_id, parent_memory_id)
    WHERE parent_memory_id IS NOT NULL;
