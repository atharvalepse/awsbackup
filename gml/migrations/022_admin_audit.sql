-- 022: admin action audit log.
--
-- The standalone admin console (akhrots.com/admin) can mutate accounts —
-- change plan/role, suspend/restore access, and DELETE users (which cascades
-- to their memories + conversations). Those are exactly the actions you want
-- an immutable trail for: who did what, to whom, when, and the before/after.
--
-- admin_audit is append-only. Every mutation endpoint writes one row inside
-- the same request. ~150 bytes/row. Admin-only (RLS admin bypass, like 015).

BEGIN;

CREATE TABLE IF NOT EXISTS admin_audit (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_id     TEXT NOT NULL,          -- admin user_id who performed it ('admin' = master key)
    actor_email  TEXT,                   -- admin's email, if known
    action       TEXT NOT NULL,          -- set_plan | set_active | delete_user | generate_invite
    target_id    TEXT,                   -- affected user_id
    target_email TEXT,                   -- affected user's email, captured at action time
    detail       JSONB                   -- {"from":..,"to":..} / arbitrary context
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_ts     ON admin_audit (ts DESC);
CREATE INDEX IF NOT EXISTS idx_admin_audit_target ON admin_audit (target_id, ts DESC);

ALTER TABLE admin_audit ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS admin_audit_self ON admin_audit;
CREATE POLICY admin_audit_self ON admin_audit
    USING (current_setting('app.is_admin', TRUE) = 'true');

GRANT SELECT, INSERT ON admin_audit TO gml_app;
GRANT USAGE ON SEQUENCE admin_audit_id_seq TO gml_app;

COMMIT;
