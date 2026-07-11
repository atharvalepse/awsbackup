-- 015: persist write-gate decision rationale.
--
-- The novelty/supersession/conflict gate (_apply_write_gate) computes a
-- similarity to the nearest active belief, compares signal tokens, and
-- picks TOUCH / SUPERSEDE / CONFLICT / INSERT — then (pre-015) discarded
-- all of that, keeping only the outcome embedded in raw_metadata links.
-- That made "why was this belief closed / touched / flagged?" unanswerable
-- without re-running the gate.
--
-- gate_decisions is an append-only audit ledger: one row per claim per
-- add_many pass. ~100 bytes/row; written in a single executemany inside
-- the same transaction as the memories insert.

BEGIN;

CREATE TABLE IF NOT EXISTS gate_decisions (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    claim_id        TEXT NOT NULL,   -- incoming claim's dedup id
    decision        TEXT NOT NULL CHECK (
        decision IN ('insert', 'touch', 'supersede', 'conflict', 'retract')
    ),
    nearest_id      TEXT,            -- nearest active neighbour considered
    similarity      REAL,            -- cosine sim to nearest at decision time
    reason          TEXT,            -- e.g. 'sim>=touch', 'signals_equal',
                                     -- 'cue', 'polarity_guard', 'no_neighbour'
    matched_signals TEXT[],          -- signal tokens that drove the decision
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- "Why is claim X in this state?" — point lookup by claim.
CREATE INDEX IF NOT EXISTS idx_gate_decisions_user_claim
    ON gate_decisions (user_id, claim_id);
-- Truth-dashboard feeds: recent decisions per tenant.
CREATE INDEX IF NOT EXISTS idx_gate_decisions_user_time
    ON gate_decisions (user_id, created_at DESC);

-- RLS — same pattern as memories (005) / entities (014)
ALTER TABLE gate_decisions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS gate_decisions_self ON gate_decisions;
CREATE POLICY gate_decisions_self ON gate_decisions
    USING (
        user_id = current_setting('app.current_user_id', TRUE)
        OR current_setting('app.is_admin', TRUE) = 'true'
    );

GRANT SELECT, INSERT ON gate_decisions TO gml_app;
GRANT USAGE ON SEQUENCE gate_decisions_id_seq TO gml_app;

COMMIT;
