-- Migration 009 — password auth for users.
--
-- Adds a password hash so users can sign up / log in with email + password
-- (issuing JWTs), in addition to the admin-issued API keys. The hash is a
-- self-describing pbkdf2 string (pbkdf2_sha256$iterations$salt$hash) produced
-- by orchestration.auth.passwords — no extra DB extension needed.
--
-- NULLable: users created by the admin key-issuance flow (no password) keep
-- working; only the email+password signup path sets it.

BEGIN;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS password_hash TEXT;

COMMENT ON COLUMN users.password_hash IS
    'pbkdf2_sha256$iterations$salt_b64$hash_b64 — set by email+password signup; NULL for key-only users.';

COMMIT;
