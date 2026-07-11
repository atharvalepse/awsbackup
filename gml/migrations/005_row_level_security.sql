-- Migration 005 — Row-Level Security for per-user data isolation.
--
-- The application sets `app.current_user_id` on every request (after auth
-- middleware resolves the API key to a user). Postgres then enforces that
-- SELECT/UPDATE/DELETE on `memories` and `user_keys` only sees rows owned
-- by that user. Even if a developer forgets a WHERE clause, data does not
-- leak.
--
-- Set `app.is_admin = 'true'` to bypass RLS (used for admin endpoints
-- that legitimately need cross-tenant access).
--
-- The `gml_app` role obeys RLS. The `postgres` superuser bypasses it
-- (BYPASSRLS is implicit for the cluster owner).

BEGIN;

ALTER TABLE memories     ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_keys    ENABLE ROW LEVEL SECURITY;

-- Drop existing policies first — idempotent re-runs are safe.
DROP POLICY IF EXISTS memories_self ON memories;
CREATE POLICY memories_self ON memories
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );

DROP POLICY IF EXISTS user_keys_self ON user_keys;
CREATE POLICY user_keys_self ON user_keys
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );

-- Grants are idempotent too.
GRANT USAGE ON SCHEMA public TO gml_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO gml_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO gml_app;

-- Future tables inherit these grants.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO gml_app;

COMMIT;
