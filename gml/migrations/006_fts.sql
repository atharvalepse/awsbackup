-- Migration 006 — Full-text search (tsvector) for BM25-style sparse retrieval.
--
-- Why we need this: the pg_trgm index from 003 is great for substring
-- LIKE queries but not for relevance-ranked search. The HybridRetriever's
-- sparse leg needs BM25-like scoring → tsvector + ts_rank_cd is the
-- canonical Postgres path.
--
-- Generated STORED column: Postgres recomputes on every UPDATE; no app
-- code needed. The setweight() weights mirror "what part of the row is
-- most relevant": content > entity/value > attribute.

BEGIN;

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(content,   '')), 'A')
        || setweight(to_tsvector('english', coalesce(entity,    '')), 'B')
        || setweight(to_tsvector('english', coalesce(value,     '')), 'B')
        || setweight(to_tsvector('english', coalesce(attribute, '')), 'C')
    ) STORED;

CREATE INDEX IF NOT EXISTS memories_content_tsv_idx
    ON memories USING GIN (content_tsv);

COMMENT ON COLUMN memories.content_tsv IS
    'Full-text vector for BM25-style sparse retrieval. Weighted: content=A, entity+value=B, attribute=C. Query via ts_rank_cd(content_tsv, plainto_tsquery(''english'', $q)).';

COMMIT;
