-- Migration 012: invite_codes table + admin user seed
-- Applied: 2026-06-09

BEGIN;

CREATE TABLE IF NOT EXISTS invite_codes (
    code           TEXT PRIMARY KEY,
    created_by     TEXT NOT NULL DEFAULT 'admin',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ,
    used_by_email  TEXT,
    used_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS invite_codes_unused
    ON invite_codes (code) WHERE used_by_email IS NULL;

-- Seed the admin user (jigneshraheja07@gmail.com)
-- ON CONFLICT: if email exists, just upgrade to admin plan
INSERT INTO users (user_id, email, password_hash, plan, quota_bytes, is_active)
VALUES (
    'usr_admin_jignesh',
    'jigneshraheja07@gmail.com',
    'pbkdf2:sha256:260000$f8ecf281cee070677c4d8cd2549bd6fc$9d8eefe9192fcbfe89f676eb677050a9007b4a4ca3bf88a544fc822ad82af7f2',
    'admin',
    107374182400,
    true
)
ON CONFLICT (email) DO UPDATE
    SET plan = 'admin',
        is_active = true,
        password_hash = EXCLUDED.password_hash,
        quota_bytes   = EXCLUDED.quota_bytes;

-- Record migration
INSERT INTO schema_migrations (filename) VALUES ('012')
ON CONFLICT DO NOTHING;

COMMIT;
