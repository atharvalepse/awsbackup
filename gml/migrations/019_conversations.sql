-- Migration 019 — conversation memory cards.
--
-- One row per captured chat turn (user prompt + AI reply): an LLM-generated
-- title + summary, plus the atomic facts extracted from the turn (which are
-- ALSO stored as normal `memories` rows so retrieval keeps working). This is
-- the gmlcore port of the MemoryBridge extension's "memory card" feature.
--
-- RLS mirrors `memories` (005/017): a user sees only its own cards; is_admin
-- bypasses. FORCE so even the table owner is subject to the policy.

BEGIN;

CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT        PRIMARY KEY,
    user_id       TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title         TEXT,
    summary       TEXT,
    user_prompt   TEXT,
    ai_response   TEXT,
    source_url    TEXT,
    source_model  TEXT,
    facts         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    fact_count    INT         NOT NULL DEFAULT 0,
    embedding     vector(384),                -- bge-small-en-v1.5, matches `memories`
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_user_time_idx
    ON conversations (user_id, created_at DESC);

-- Row-level security — same self-or-admin shape as `memories`.
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations FORCE  ROW LEVEL SECURITY;
DROP POLICY IF EXISTS conversations_self ON conversations;
CREATE POLICY conversations_self ON conversations
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );

COMMIT;
