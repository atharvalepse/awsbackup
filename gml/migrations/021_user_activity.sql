-- 021: per-request user activity ledger powering the admin dashboard.
--
-- Pre-021 the system kept no presence/usage signal at all: there was no
-- last_seen, no session record, no way to answer "who is active right now?"
-- or "how long do people actually spend in the app?".
--
-- user_activity is an append-only telemetry ledger. One row is written per
-- user per ~60s *active* window — the API auth middleware throttles + fires
-- the insert fire-and-forget, so an idle user costs nothing and a busy user
-- costs one ~80-byte row a minute. From this single stream we derive:
--   * live presence      — DISTINCT user_id with ts > now() - 5 min
--   * DAU / WAU / MAU     — DISTINCT user_id over 1 / 7 / 30 day windows
--   * time-spent          — sessionize per user (gap > 30 min = new session),
--                           sum(max(ts) - min(ts)) across a user's sessions
--   * request volume      — count(*) over a window
--   * recent-activity feed
--
-- Mirrors the gate_decisions (015) audit-ledger shape: BIGSERIAL id, RLS with
-- the standard app.is_admin bypass, explicit gml_app grants. No FK to users:
-- like gate_decisions this is a loose append-only log, and the middleware
-- only ever records already-resolved (hence existing) user ids.

BEGIN;

CREATE TABLE IF NOT EXISTS user_activity (
    id        BIGSERIAL PRIMARY KEY,
    user_id   TEXT NOT NULL,
    ts        TIMESTAMPTZ NOT NULL DEFAULT now(),
    method    TEXT,            -- HTTP method of the touch (GET/POST/...)
    path      TEXT,            -- request path (truncated to 256 chars upstream)
    ip        TEXT             -- client ip, best-effort
);

-- Per-user timeline: presence checks, sessionization, per-user drill-down.
CREATE INDEX IF NOT EXISTS idx_user_activity_user_ts
    ON user_activity (user_id, ts DESC);
-- Global recency: DAU/WAU/MAU windows + the recent-activity feed.
CREATE INDEX IF NOT EXISTS idx_user_activity_ts
    ON user_activity (ts DESC);

-- RLS — same admin-bypass pattern as gate_decisions (015) / memories (005).
-- Every reader/writer here goes through the store's app.is_admin bypass.
ALTER TABLE user_activity ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS user_activity_self ON user_activity;
CREATE POLICY user_activity_self ON user_activity
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );

GRANT SELECT, INSERT ON user_activity TO gml_app;
GRANT USAGE ON SEQUENCE user_activity_id_seq TO gml_app;

COMMIT;
