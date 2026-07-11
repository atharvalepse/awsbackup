-- Migration 001 — enable Postgres extensions GML depends on.
--
-- Run order: 1st. Other migrations assume these exist.
--
-- Apply:
--   psql "$GML_DATABASE_URL" -f migrations/001_extensions.sql
--
-- Cloud SQL note: extensions must be enabled in the Cloud SQL console first
-- (Database → Extensions). The CREATE EXTENSION calls below will then succeed.
-- On self-hosted Postgres there's no console step.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector — dense retrieval
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- substring similarity (list_memories filters)
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid() (future-proofing)

-- A small bookkeeping table so apply_all.sh can detect already-applied
-- migrations and skip them. Idempotent: created only on first run.
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    checksum    TEXT   -- optional SHA256 of the file; populated by apply_all.sh
);

COMMIT;
