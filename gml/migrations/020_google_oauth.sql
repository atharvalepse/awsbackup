-- Migration 020 — Google (OAuth) sign-in.
--
-- Records the Google account subject id ("sub") on the user row so we can
-- (a) audit which accounts are Google-linked and (b) detect the same Google
-- identity even if its email display changes. Accounts are still matched by
-- verified email at sign-in; google_sub is supplementary, hence NULLable
-- (every pre-existing user, and all email+password users, keep working).
--
-- Google-only accounts have password_hash NULL (added in 009) and google_sub
-- set; email+password accounts have password_hash set and google_sub NULL
-- until they first use the Google button.

BEGIN;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS google_sub TEXT;

-- A given Google identity maps to at most one account. Partial unique index
-- so the many NULLs (non-Google users) don't collide.
CREATE UNIQUE INDEX IF NOT EXISTS users_google_sub_key
    ON users (google_sub) WHERE google_sub IS NOT NULL;

COMMENT ON COLUMN users.google_sub IS
    'Google Identity "sub" claim — set when the account signs in with Google; NULL otherwise.';

COMMIT;
