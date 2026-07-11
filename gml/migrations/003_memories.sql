-- Migration 003 — the memories table + non-FTS indexes.
--
-- Vector dim is locked to 384 (bge-small-en-v1.5). If you ever swap to a
-- different embedder (e.g. bge-large = 1024), DROP and recreate the
-- embedding column — pgvector does NOT auto-resize, and the HNSW index
-- is dim-specific.

BEGIN;

CREATE TABLE IF NOT EXISTS memories (
    id               TEXT        PRIMARY KEY,
    user_id          TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content          TEXT        NOT NULL,
    entity           TEXT,
    attribute        TEXT,
    value            TEXT,
    source           TEXT        NOT NULL DEFAULT 'conversation',
    authority_score  REAL        NOT NULL DEFAULT 0.7,
    pinned           BOOLEAN     NOT NULL DEFAULT FALSE,
    "timestamp"      TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_metadata     JSONB,
    summary_short    TEXT,
    embedding        vector(384),
    byte_size        INTEGER     NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The most-common filter: "give me memory X for user Y"
CREATE INDEX IF NOT EXISTS memories_user_id_idx
    ON memories(user_id);

-- Entity-scoped browse: /api/memories?entity=payments
CREATE INDEX IF NOT EXISTS memories_user_entity_idx
    ON memories(user_id, entity)
    WHERE entity IS NOT NULL;

-- Recency-ordered queries (e.g. "show me the last 50 memories")
CREATE INDEX IF NOT EXISTS memories_user_time_idx
    ON memories(user_id, "timestamp" DESC);

-- Substring search for the list_memories filter. NOT used for BM25 ranking;
-- migration 006 adds the real FTS column.
CREATE INDEX IF NOT EXISTS memories_content_trgm_idx
    ON memories USING GIN (content gin_trgm_ops);

-- Vector index. HNSW is right for our access pattern (sub-100ms top-K).
-- m=16, ef_construction=64 is a good default for <1M vectors. If you grow
-- past that, raise m to 32 for better recall at index-build cost.
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMIT;
