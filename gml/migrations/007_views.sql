-- Migration 007 — convenience views.
--
-- `user_quota_status` is the canonical read path for the UI's "you're at
-- X% of your 1 GB" UI hint and for the /api/me/quota endpoint.

BEGIN;

CREATE OR REPLACE VIEW user_quota_status AS
SELECT
    u.user_id,
    u.plan,
    u.quota_bytes,
    u.bytes_used,
    CASE
        WHEN u.quota_bytes = 0 THEN 0.0
        ELSE LEAST(1.0, u.bytes_used::float / u.quota_bytes::float)
    END AS pct_used,
    u.warned_at_90pct,
    (SELECT count(*) FROM memories m WHERE m.user_id = u.user_id) AS memory_count
FROM users u;

GRANT SELECT ON user_quota_status TO gml_app;

COMMIT;
