-- Migration 008 — first-class AAL columns on memories.
--
-- The canonical persisted format is now AAL = {simplemem, sjson}. Both views
-- of every memory are stored as dedicated columns so:
--   * ``aal_simplemem`` is a real TEXT column, indexable by trgm/tsvector
--     for content-side search (the existing content_tsv index from 006
--     covers it already because content_tsv reads `content`, and our
--     AAL write path sets content := simplemem).
--   * ``aal_sjson`` is JSONB so we can use Postgres path operators
--     (``->``, ``->>``, ``@>``) to query structured triples without
--     parsing JSON on every read.
--
-- Both columns are NULLable so legacy rows that pre-date AAL keep working.
-- The application reads ``aal_simplemem`` if set, falling back to ``content``.
-- Similarly ``aal_sjson`` falls back to raw_metadata->'sjson' (if present)
-- or to the {entity, attribute, value} triple.

BEGIN;

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS aal_simplemem TEXT,
    ADD COLUMN IF NOT EXISTS aal_sjson JSONB;

-- GIN index on the JSONB column so containment queries (e.g. find every
-- memory whose sjson subject = "payments") are fast even at scale.
-- jsonb_path_ops is leaner than the default jsonb_ops and matches our
-- read pattern (we only do @> containment, not arbitrary path lookups).
CREATE INDEX IF NOT EXISTS memories_aal_sjson_idx
    ON memories USING GIN (aal_sjson jsonb_path_ops);

-- Optional: a partial index on (sjson subject) for the most common
-- structured filter. The expression has to be IMMUTABLE for an index
-- expression; jsonb_extract_path_text is immutable.
CREATE INDEX IF NOT EXISTS memories_aal_subject_idx
    ON memories ((aal_sjson ->> 'subject'))
    WHERE aal_sjson IS NOT NULL;

COMMENT ON COLUMN memories.aal_simplemem IS
    'Canonical AAL view: one-line natural sentence. Mirrors content for AAL-written rows.';
COMMENT ON COLUMN memories.aal_sjson IS
    'Canonical AAL view: structured triple {subject, verb, object, time, negated, confidence, category, ...}. NULL for legacy rows.';

COMMIT;
