-- 013_bitemporal.sql — real bitemporal claims (additive; is_latest kept in sync).
--
-- A memory is a bitemporal claim:
--   valid_from  world time the fact began to hold (defaults to the claim's
--               own "timestamp" — when the user said it)
--   valid_to    NULL = currently believed. Supersession CLOSES the interval
--               (sets valid_to + superseded_by in raw_metadata); nothing is
--               ever deleted.
--   tx_time     system time we learned the claim (created_at for old rows)
--
-- is_latest stays as a physical column synced by trigger:
--   is_latest := (valid_to IS NULL)
-- so every existing reader (both retriever legs filter on it) keeps working
-- unchanged while readers migrate to validity-window filters. The trigger is
-- the single source of sync — writers set valid_to only.
--
-- Downgrade: migrations/downgrades/013_bitemporal_down.sql (drops trigger,
-- indexes and columns; is_latest retains its values, so pre-013 readers are
-- whole again).

BEGIN;

-- ── 0. Ledger repair ────────────────────────────────────────────────────
-- 009–012 were applied to production by hand and never recorded (012's own
-- INSERT referenced a nonexistent "version" column). Record them so the
-- ledger matches physical reality. No-op on databases where apply_all.sh
-- recorded them properly.
INSERT INTO schema_migrations (filename)
SELECT f FROM unnest(ARRAY[
    '009_user_passwords.sql',
    '010_is_latest.sql',
    '011_parent_memory_id.sql',
    '012_invite_codes.sql'
]) AS t(f)
WHERE EXISTS (  -- only when the migration's artifact is physically present
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'memories' AND column_name = 'is_latest'
)
ON CONFLICT (filename) DO NOTHING;

-- ── 1. Columns (additive, nullable first so the ALTER is instant) ───────
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS valid_to   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS tx_time    TIMESTAMPTZ;

-- ── 2. Backfill ─────────────────────────────────────────────────────────
-- (table is ~1k rows today; if this ever runs on a table large enough to
-- matter, convert to batched UPDATEs on id ranges.)
UPDATE memories SET tx_time = created_at WHERE tx_time IS NULL;
UPDATE memories SET valid_from = "timestamp" WHERE valid_from IS NULL;

-- Superseded rows: close the interval at the superseding claim's tx_time
-- when the forward link exists, else at the row's own tx_time.
UPDATE memories m
SET valid_to = COALESCE(
        (SELECT s.created_at FROM memories s
         WHERE s.id = m.raw_metadata->>'superseded_by'
           AND s.user_id = m.user_id),
        m.created_at)
WHERE m.valid_to IS NULL AND NOT m.is_latest;

-- ── 3. Constraints + defaults ───────────────────────────────────────────
ALTER TABLE memories
    ALTER COLUMN valid_from SET DEFAULT now(),
    ALTER COLUMN valid_from SET NOT NULL,
    ALTER COLUMN tx_time    SET DEFAULT now(),
    ALTER COLUMN tx_time    SET NOT NULL;

-- A closed interval must not end before it starts (equal is allowed: a
-- claim can be believed and retracted in the same instant by a correction).
ALTER TABLE memories
    ADD CONSTRAINT memories_valid_interval
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
    NOT VALID;  -- existing rows validated separately to avoid a long lock
ALTER TABLE memories VALIDATE CONSTRAINT memories_valid_interval;

-- ── 4. is_latest sync trigger ───────────────────────────────────────────
-- One source of truth: writers manage valid_to; is_latest is derived.
CREATE OR REPLACE FUNCTION memories_sync_is_latest() RETURNS trigger AS $$
BEGIN
    NEW.is_latest := (NEW.valid_to IS NULL);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_memories_sync_is_latest ON memories;
CREATE TRIGGER trg_memories_sync_is_latest
    BEFORE INSERT OR UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION memories_sync_is_latest();

-- ── 5. Indexes for the two real query patterns ──────────────────────────
-- Hot path: current beliefs for one tenant (every recall query).
-- NOTE on a large live table use CREATE INDEX CONCURRENTLY (outside a txn).
CREATE INDEX IF NOT EXISTS idx_memories_current
    ON memories (user_id, entity, attribute)
    WHERE valid_to IS NULL;

-- Time travel: valid_from <= :as_of AND (valid_to IS NULL OR valid_to > :as_of)
CREATE INDEX IF NOT EXISTS idx_memories_validity
    ON memories (user_id, valid_from, valid_to);

COMMIT;
