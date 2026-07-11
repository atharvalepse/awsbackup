-- Downgrade for 014_entity_resolution.sql.
-- Apply manually:
--   psql "$GML_DATABASE_URL" -v ON_ERROR_STOP=1 \
--     -f migrations/downgrades/014_entity_resolution_down.sql
--
-- memories.entity (display text) is untouched by 014, so dropping
-- entity_id returns readers to the pre-014 contract losslessly.

BEGIN;

DROP INDEX IF EXISTS idx_memories_entity_id;
ALTER TABLE memories DROP COLUMN IF EXISTS entity_id;

DROP TABLE IF EXISTS entity_merge_candidates;
DROP TABLE IF EXISTS entity_aliases;
DROP TABLE IF EXISTS entities;

DELETE FROM schema_migrations WHERE filename = '014_entity_resolution.sql';

COMMIT;
