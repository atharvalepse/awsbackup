-- Downgrade for 013_bitemporal.sql.
--
-- Lives under downgrades/ so apply_all.sh's [0-9][0-9][0-9]_*.sql glob can
-- never auto-apply it. Apply manually:
--   psql "$GML_DATABASE_URL" -v ON_ERROR_STOP=1 \
--     -f migrations/downgrades/013_bitemporal_down.sql
--
-- Safe because is_latest remained a physical column kept in sync by the
-- trigger for the whole time 013 was live — dropping the bitemporal columns
-- returns readers to exactly the pre-013 contract with no data loss in
-- pre-013 fields. (valid_from/valid_to/tx_time values are lost, which is
-- what a downgrade means; superseded_by links in raw_metadata survive.)

BEGIN;

DROP TRIGGER IF EXISTS trg_memories_sync_is_latest ON memories;
DROP FUNCTION IF EXISTS memories_sync_is_latest();

DROP INDEX IF EXISTS idx_memories_current;
DROP INDEX IF EXISTS idx_memories_validity;

ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_valid_interval;

ALTER TABLE memories
    DROP COLUMN IF EXISTS valid_from,
    DROP COLUMN IF EXISTS valid_to,
    DROP COLUMN IF EXISTS tx_time;

DELETE FROM schema_migrations WHERE filename = '013_bitemporal.sql';

COMMIT;
