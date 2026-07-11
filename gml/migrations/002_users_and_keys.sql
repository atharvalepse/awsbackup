-- Migration 002 — users + user_keys.
--
-- Quota fields live on the `users` row. The byte counter is maintained
-- automatically by a trigger added in 004; never write to bytes_used directly.

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    user_id          TEXT        PRIMARY KEY,
    email            TEXT        UNIQUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    plan             TEXT        NOT NULL DEFAULT 'free'
                                 CHECK (plan IN ('free', 'pro', 'team', 'admin')),
    -- Free tier = 1 GiB. Bumping the plan can raise this via UPDATE.
    quota_bytes      BIGINT      NOT NULL DEFAULT 1073741824,
    bytes_used       BIGINT      NOT NULL DEFAULT 0,
    -- Soft-cap state: set when bytes_used first crosses 90% of quota.
    -- Cleared when the user drops back below 90% (so a future re-cross re-warns).
    warned_at_90pct  BOOLEAN     NOT NULL DEFAULT FALSE,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS user_keys (
    key            TEXT        PRIMARY KEY,
    user_id        TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    label          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at   TIMESTAMPTZ,
    is_active      BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS user_keys_user_id_idx ON user_keys(user_id);
CREATE INDEX IF NOT EXISTS user_keys_active_idx  ON user_keys(is_active) WHERE is_active;

COMMIT;
