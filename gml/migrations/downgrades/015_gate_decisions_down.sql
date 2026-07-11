-- Downgrade for 015_gate_decisions.sql.
-- Apply manually:
--   psql "$GML_DATABASE_URL" -v ON_ERROR_STOP=1 \
--     -f migrations/downgrades/015_gate_decisions_down.sql
--
-- gate_decisions is a write-only audit ledger; nothing reads it on the
-- hot path, so dropping it is lossless for belief state.

BEGIN;

DROP TABLE IF EXISTS gate_decisions;

COMMIT;
