-- Downgrade for 016_session_id.sql.
-- Apply manually:
--   psql "$GML_DATABASE_URL" -v ON_ERROR_STOP=1 \
--     -f migrations/downgrades/016_session_id_down.sql
--
-- raw_metadata retains the session_id key, so dropping the column is
-- lossless for pre-016 readers.

BEGIN;

DROP INDEX IF EXISTS idx_memories_session;
ALTER TABLE memories DROP COLUMN IF EXISTS session_id;

COMMIT;
