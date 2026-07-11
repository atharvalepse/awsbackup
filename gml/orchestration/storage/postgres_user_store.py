"""PostgresUserKeyStore — async user/key store backed by Postgres.

Same async API as the JSONL ``UserKeyStore``: ``lookup``, ``issue``,
``revoke``, ``list_users``, ``by_user_id``. The admin endpoints in
``server.py`` consume this transparently — set ``GML_STORAGE_BACKEND=postgres``
and they switch over.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from orchestration.observability.logging import StructuredLogger
from orchestration.users import UserRecord

if TYPE_CHECKING:
    import asyncpg


slog = StructuredLogger("user_store.postgres")


class PostgresUserKeyStore:
    """asyncpg-backed user + key store.

    Mirrors the public surface of :class:`orchestration.users.UserKeyStore`,
    except every method is ``async`` (callers that previously called the
    JSONL store sync now need an ``await`` — server.py middleware does this).
    """

    def __init__(self, pool: "asyncpg.Pool") -> None:
        self.pool = pool

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    async def lookup(self, key: str) -> UserRecord | None:
        if not key:
            return None
        async with self.pool.acquire() as conn:
            # The key→user lookup must bypass RLS (it reads across all keys to
            # find a match, before any user context exists). set_config(...,
            # is_local=true) only persists for the current transaction, so the
            # bypass and the SELECT MUST share one transaction — otherwise, on a
            # pooled autocommit connection, the GUC resets between statements,
            # RLS hides every user_keys row, and all user auth 401s.
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                row = await conn.fetchrow(
                    """
                    SELECT k.key, k.user_id, k.label, k.created_at
                    FROM user_keys k
                    JOIN users u ON u.user_id = k.user_id
                    WHERE k.key = $1 AND k.is_active AND u.is_active
                    """,
                    key,
                )
                if row is None:
                    return None
                # Touch last_used_at so we can see active vs dormant keys later.
                await conn.execute(
                    "UPDATE user_keys SET last_used_at = now() WHERE key = $1",
                    key,
                )
        return UserRecord(
            key=row["key"],
            user_id=row["user_id"],
            created_at=row["created_at"].isoformat()
                if isinstance(row["created_at"], datetime)
                else str(row["created_at"]),
            label=row["label"],
        )

    async def issue(self, user_id: str, label: str | None = None) -> UserRecord:
        if not user_id:
            raise ValueError("user_id is required")
        key = "gml_" + secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                # Idempotently create the user row (default 1 GiB quota).
                await conn.execute(
                    """
                    INSERT INTO users (user_id, plan, quota_bytes)
                    VALUES ($1, 'free', 1073741824)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    user_id,
                )
                await conn.execute(
                    """
                    INSERT INTO user_keys (key, user_id, label, created_at)
                    VALUES ($1, $2, $3, $4)
                    """,
                    key, user_id, label, now,
                )
        slog.info(event="pg_key_issued", user_id=user_id, label=label)
        return UserRecord(
            key=key, user_id=user_id, created_at=now.isoformat(), label=label
        )

    async def revoke(self, key: str) -> bool:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                # Soft-revoke (is_active = false) rather than DELETE so we
                # keep an audit trail of when keys were retired.
                result = await conn.execute(
                    "UPDATE user_keys SET is_active = FALSE WHERE key = $1 AND is_active",
                    key,
                )
        revoked = result.endswith(" 1")
        if revoked:
            slog.info(event="pg_key_revoked", key_prefix=key[:8])
        return revoked

    async def list_users(self) -> list[UserRecord]:
        async with self.pool.acquire() as conn:
            # Same transaction-scoped is_admin bypass as lookup() — the GUC and
            # the SELECT must share a transaction or RLS hides the rows.
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                rows = await conn.fetch(
                    """
                    SELECT key, user_id, label, created_at
                    FROM user_keys
                    WHERE is_active
                    ORDER BY created_at DESC
                    """
                )
        return [
            UserRecord(
                key=r["key"],
                user_id=r["user_id"],
                created_at=(r["created_at"].isoformat()
                            if isinstance(r["created_at"], datetime)
                            else str(r["created_at"])),
                label=r["label"],
            )
            for r in rows
        ]

    async def by_user_id(self, user_id: str):
        """All active keys for one user."""
        for rec in await self.list_users():
            if rec.user_id == user_id:
                yield rec

    # ----------------------------------------------------------------------
    # Email + password auth (used by /auth/signup, /auth/login, /auth/me)
    # ----------------------------------------------------------------------

    async def create_user_with_password(
        self, user_id: str, email: str, password_hash: str
    ) -> bool:
        """Create a user row with a password. Returns False if the email is
        already taken (UNIQUE violation), True on success."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                try:
                    await conn.execute(
                        """
                        INSERT INTO users (user_id, email, password_hash, plan, quota_bytes)
                        VALUES ($1, $2, $3, 'free', 1073741824)
                        """,
                        user_id, email, password_hash,
                    )
                except Exception as exc:  # asyncpg UniqueViolation -> sqlstate 23505
                    if getattr(exc, "sqlstate", None) == "23505":
                        return False
                    raise
        slog.info(event="pg_user_created", user_id=user_id)
        return True

    async def get_user_auth_by_email(self, email: str) -> dict | None:
        """Return {user_id, password_hash, is_active} for an email, or None."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                row = await conn.fetchrow(
                    "SELECT user_id, password_hash, is_active FROM users WHERE email = $1",
                    email,
                )
        return dict(row) if row else None

    async def create_oauth_user(
        self, user_id: str, email: str, google_sub: str | None = None
    ) -> bool:
        """Create a user row for a Google (OAuth) sign-up. ``password_hash`` is
        left NULL — these accounts authenticate via Google only. Returns False
        if the email is already taken (UNIQUE violation), True on success."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                try:
                    await conn.execute(
                        """
                        INSERT INTO users (user_id, email, google_sub, plan, quota_bytes)
                        VALUES ($1, $2, $3, 'free', 1073741824)
                        """,
                        user_id, email, google_sub,
                    )
                except Exception as exc:  # asyncpg UniqueViolation -> sqlstate 23505
                    if getattr(exc, "sqlstate", None) == "23505":
                        return False
                    raise
        slog.info(event="pg_oauth_user_created", user_id=user_id)
        return True

    async def link_google_sub(self, user_id: str, google_sub: str) -> None:
        """Record the Google ``sub`` on an existing account the first time it
        signs in with Google (e.g. an email+password account that later uses
        the Google button). Best-effort; no-op if already set."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                await conn.execute(
                    "UPDATE users SET google_sub = $2 "
                    "WHERE user_id = $1 AND google_sub IS NULL",
                    user_id, google_sub,
                )


    # ------------------------------------------------------------------
    # Invite code methods
    # ------------------------------------------------------------------

    async def generate_invite_code(self, created_by: str = "admin") -> str:
        """Generate a new AKH-XXXX invite code and persist it. Returns the code."""
        import secrets as _sec
        code = "AKH-" + _sec.token_hex(4).upper()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.is_admin', 'true', true)"
                )
                await conn.execute(
                    """INSERT INTO invite_codes (code, created_by) VALUES ($1, $2)""",
                    code, created_by,
                )
        return code

    async def validate_and_use_invite_code(self, code: str, email: str) -> bool:
        """
        Atomically validates and marks an invite code as used.
        Returns True on success, False if the code is invalid/already used/expired.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.is_admin', 'true', true)"
                )
                row = await conn.fetchrow(
                    """
                    SELECT code, used_by_email, expires_at
                    FROM   invite_codes
                    WHERE  code = $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    code,
                )
                if row is None:
                    return False
                if row["used_by_email"] is not None:
                    return False
                from datetime import datetime, timezone as _tz
                if row["expires_at"] and row["expires_at"] < datetime.now(_tz.utc):
                    return False
                await conn.execute(
                    """
                    UPDATE invite_codes
                    SET    used_by_email = $1, used_at = now()
                    WHERE  code = $2
                    """,
                    email, code,
                )
        return True

    async def list_invite_codes(self) -> list:
        """Return all invite codes ordered newest-first."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.is_admin', 'true', true)"
                )
                rows = await conn.fetch(
                    """
                    SELECT code, created_by, created_at, used_by_email, used_at, expires_at
                    FROM   invite_codes
                    ORDER  BY created_at DESC
                    """
                )
        return [dict(r) for r in rows]

    async def get_user(self, user_id: str) -> dict | None:
        """Public profile fields for one user (for /auth/me)."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                row = await conn.fetchrow(
                    "SELECT user_id, email, plan FROM users WHERE user_id = $1",
                    user_id,
                )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Activity tracking (powers the admin dashboard, migration 021)
    # ------------------------------------------------------------------

    async def record_activity(
        self,
        user_id: str,
        method: str | None = None,
        path: str | None = None,
        ip: str | None = None,
    ) -> None:
        """Append one activity touch to ``user_activity``.

        Called fire-and-forget from the API auth middleware (throttled to
        ~once/user/60s upstream), so it must stay cheap and never raise into
        the request path — the middleware swallows any error this surfaces.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                await conn.execute(
                    """
                    INSERT INTO user_activity (user_id, method, path, ip)
                    VALUES ($1, $2, $3, $4)
                    """,
                    user_id,
                    (method or None),
                    (path or None) if path is None else path[:256],
                    (ip or None),
                )

    async def activity_dashboard(self) -> dict:
        """Everything the admin dashboard renders, computed server-side.

        Returns a JSON-ready dict with four sections:

        * ``overview``   — headline counters (online now, DAU/WAU/MAU, totals).
        * ``users``      — one row per account (incl. zero-activity ones) with
          last-seen, sessionized time-spent (gap > 30 min = a new session),
          today's time, session + event counts, plan and signup date.
        * ``recent``     — the newest 50 activity touches for the live feed.
        * ``timeseries`` — hourly event / distinct-user buckets for the last
          24h (drives the dashboard sparkline).

        A "session" is a run of one user's touches with < 30 min between
        consecutive touches; its duration is ``max(ts) - min(ts)`` within the
        run. Single-touch sessions therefore contribute 0 s, which is honest:
        we only know they pinged once.
        """
        import datetime as _dt

        def _iso(v):
            return v.isoformat() if isinstance(v, _dt.datetime) else (str(v) if v is not None else None)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")

                overview_row = await conn.fetchrow(
                    """
                    SELECT
                      (SELECT count(*) FROM users)                                  AS total_users,
                      (SELECT count(*) FROM users WHERE is_active)                  AS active_accounts,
                      (SELECT count(*) FROM users WHERE plan = 'admin')             AS admins,
                      (SELECT count(*) FROM users
                         WHERE created_at > now() - interval '7 days')              AS new_7d,
                      (SELECT count(DISTINCT user_id) FROM user_activity
                         WHERE ts > now() - interval '5 minutes')                   AS online_now,
                      (SELECT count(DISTINCT user_id) FROM user_activity
                         WHERE ts > now() - interval '24 hours')                    AS dau,
                      (SELECT count(DISTINCT user_id) FROM user_activity
                         WHERE ts > now() - interval '7 days')                      AS wau,
                      (SELECT count(DISTINCT user_id) FROM user_activity
                         WHERE ts > now() - interval '30 days')                     AS mau,
                      (SELECT count(*) FROM user_activity
                         WHERE ts > now() - interval '24 hours')                    AS events_24h
                    """
                )

                # Per-user rollup with sessionized time-spent. Users with no
                # activity yet still appear (LEFT JOIN), sorted most-recent first.
                user_rows = await conn.fetch(
                    """
                    WITH marked AS (
                        SELECT
                            user_id, ts,
                            CASE
                              WHEN lag(ts) OVER w IS NULL
                                OR ts - lag(ts) OVER w > interval '30 minutes'
                              THEN 1 ELSE 0
                            END AS new_sess
                        FROM user_activity
                        WINDOW w AS (PARTITION BY user_id ORDER BY ts)
                    ),
                    sessioned AS (
                        SELECT user_id, ts,
                               sum(new_sess) OVER (
                                   PARTITION BY user_id ORDER BY ts
                                   ROWS UNBOUNDED PRECEDING
                               ) AS sess_no
                        FROM marked
                    ),
                    sess AS (
                        SELECT user_id, sess_no,
                               max(ts) - min(ts) AS dur,
                               max(ts)           AS sess_end,
                               count(*)          AS touches
                        FROM sessioned
                        GROUP BY user_id, sess_no
                    ),
                    agg AS (
                        SELECT
                            user_id,
                            count(*)                                          AS session_count,
                            coalesce(sum(extract(epoch FROM dur)), 0)::bigint AS total_seconds,
                            coalesce(sum(
                                CASE WHEN sess_end > now() - interval '24 hours'
                                     THEN extract(epoch FROM dur) ELSE 0 END
                            ), 0)::bigint                                     AS seconds_24h,
                            coalesce(sum(touches), 0)                         AS event_count,
                            max(sess_end)                                     AS last_seen
                        FROM sess
                        GROUP BY user_id
                    )
                    SELECT
                        u.user_id, u.email, u.plan, u.created_at, u.is_active,
                        coalesce(a.session_count, 0) AS session_count,
                        coalesce(a.total_seconds, 0) AS total_seconds,
                        coalesce(a.seconds_24h, 0)   AS seconds_24h,
                        coalesce(a.event_count, 0)   AS event_count,
                        a.last_seen
                    FROM users u
                    LEFT JOIN agg a USING (user_id)
                    ORDER BY a.last_seen DESC NULLS LAST, u.created_at DESC
                    """
                )

                recent_rows = await conn.fetch(
                    """
                    SELECT a.user_id, u.email, a.method, a.path, a.ts
                    FROM user_activity a
                    LEFT JOIN users u USING (user_id)
                    ORDER BY a.ts DESC
                    LIMIT 50
                    """
                )

                series_rows = await conn.fetch(
                    """
                    SELECT date_trunc('hour', ts) AS bucket,
                           count(*)               AS events,
                           count(DISTINCT user_id) AS users
                    FROM user_activity
                    WHERE ts > now() - interval '24 hours'
                    GROUP BY 1
                    ORDER BY 1
                    """
                )

        return {
            "overview": {k: int(v) for k, v in dict(overview_row).items()},
            "users": [
                {
                    "user_id": r["user_id"],
                    "email": r["email"],
                    "plan": r["plan"],
                    "created_at": _iso(r["created_at"]),
                    "is_active": r["is_active"],
                    "session_count": int(r["session_count"]),
                    "total_seconds": int(r["total_seconds"]),
                    "seconds_24h": int(r["seconds_24h"]),
                    "event_count": int(r["event_count"]),
                    "last_seen": _iso(r["last_seen"]),
                }
                for r in user_rows
            ],
            "recent": [
                {
                    "user_id": r["user_id"],
                    "email": r["email"],
                    "method": r["method"],
                    "path": r["path"],
                    "ts": _iso(r["ts"]),
                }
                for r in recent_rows
            ],
            "timeseries": [
                {
                    "bucket": _iso(r["bucket"]),
                    "events": int(r["events"]),
                    "users": int(r["users"]),
                }
                for r in series_rows
            ],
        }

    # ------------------------------------------------------------------
    # Admin user management (admin console, migrations 021/022)
    # ------------------------------------------------------------------

    VALID_PLANS = ("free", "pro", "team", "admin")

    async def count_active_admins(self, exclude_user_id: str | None = None) -> int:
        """How many active admins exist (optionally excluding one user).

        Used to refuse demoting/suspending/deleting the last admin, so the
        console can never lock everyone out of administration.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                row = await conn.fetchrow(
                    """
                    SELECT count(*) AS n FROM users
                    WHERE plan = 'admin' AND is_active
                      AND ($1::text IS NULL OR user_id <> $1)
                    """,
                    exclude_user_id,
                )
        return int(row["n"]) if row else 0

    async def set_user_plan(self, user_id: str, plan: str) -> dict | None:
        """Set a user's plan/role. Returns the updated public profile, or None
        if the user doesn't exist. Raises ValueError on an invalid plan."""
        if plan not in self.VALID_PLANS:
            raise ValueError(f"invalid plan: {plan!r}")
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                row = await conn.fetchrow(
                    "UPDATE users SET plan = $2 WHERE user_id = $1 "
                    "RETURNING user_id, email, plan, is_active",
                    user_id, plan,
                )
        return dict(row) if row else None

    async def set_user_active(self, user_id: str, is_active: bool) -> dict | None:
        """Suspend (is_active=false) or restore (true) a user. Returns the
        updated profile, or None if the user doesn't exist."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                row = await conn.fetchrow(
                    "UPDATE users SET is_active = $2 WHERE user_id = $1 "
                    "RETURNING user_id, email, plan, is_active",
                    user_id, is_active,
                )
        return dict(row) if row else None

    async def delete_user(self, user_id: str) -> bool:
        """Hard-delete a user and everything they own, in one transaction.

        ``conversations``, ``memories`` and ``user_keys`` cascade off the
        users FK automatically. ``entities`` does NOT cascade (and its aliases/
        merge-candidates cascade off *entities*), so we delete those first; the
        loose logs (``user_activity``, ``gate_decisions``) carry no FK and are
        cleaned for hygiene. Returns True if a users row was removed.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                # entities → cascades entity_aliases + entity_merge_candidates
                await conn.execute("DELETE FROM entities WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM user_activity WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM gate_decisions WHERE user_id = $1", user_id)
                # users → cascades conversations, memories, user_keys
                result = await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
        return result.endswith(" 1")

    async def get_user_detail(self, user_id: str) -> dict | None:
        """Rich per-user profile for the console detail drawer: identity,
        plan/status, content counts, time-spent and recent activity."""
        import datetime as _dt

        def _iso(v):
            return v.isoformat() if isinstance(v, _dt.datetime) else (str(v) if v is not None else None)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                prof = await conn.fetchrow(
                    """
                    SELECT user_id, email, plan, is_active, created_at,
                           (password_hash IS NOT NULL) AS has_password,
                           (google_sub   IS NOT NULL)  AS has_google,
                           bytes_used, quota_bytes
                    FROM users WHERE user_id = $1
                    """,
                    user_id,
                )
                if prof is None:
                    return None
                counts = await conn.fetchrow(
                    """
                    SELECT
                      (SELECT count(*) FROM memories      WHERE user_id = $1) AS memory_count,
                      (SELECT count(*) FROM conversations  WHERE user_id = $1) AS conversation_count,
                      (SELECT count(*) FROM user_keys      WHERE user_id = $1 AND is_active) AS key_count,
                      (SELECT count(*) FROM user_activity  WHERE user_id = $1) AS event_count,
                      (SELECT max(ts)  FROM user_activity  WHERE user_id = $1) AS last_seen
                    """,
                    user_id,
                )
                recent = await conn.fetch(
                    "SELECT method, path, ts FROM user_activity WHERE user_id = $1 "
                    "ORDER BY ts DESC LIMIT 20",
                    user_id,
                )
        return {
            "user_id": prof["user_id"],
            "email": prof["email"],
            "plan": prof["plan"],
            "is_active": prof["is_active"],
            "created_at": _iso(prof["created_at"]),
            "has_password": prof["has_password"],
            "has_google": prof["has_google"],
            "bytes_used": int(prof["bytes_used"]),
            "quota_bytes": int(prof["quota_bytes"]),
            "memory_count": int(counts["memory_count"]),
            "conversation_count": int(counts["conversation_count"]),
            "key_count": int(counts["key_count"]),
            "event_count": int(counts["event_count"]),
            "last_seen": _iso(counts["last_seen"]),
            "recent": [
                {"method": r["method"], "path": r["path"], "ts": _iso(r["ts"])}
                for r in recent
            ],
        }

    async def record_admin_audit(
        self,
        actor_id: str,
        actor_email: str | None,
        action: str,
        target_id: str | None,
        target_email: str | None,
        detail: dict | None = None,
    ) -> None:
        """Append one row to the admin audit log. Best-effort within the
        caller's request; never the reason a mutation fails."""
        import json as _json
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                await conn.execute(
                    """
                    INSERT INTO admin_audit
                        (actor_id, actor_email, action, target_id, target_email, detail)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    actor_id, actor_email, action, target_id, target_email,
                    _json.dumps(detail or {}),
                )

    async def list_admin_audit(self, limit: int = 100) -> list[dict]:
        """Recent admin audit entries, newest first."""
        import datetime as _dt
        limit = max(1, min(int(limit), 500))
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                rows = await conn.fetch(
                    "SELECT ts, actor_id, actor_email, action, target_id, target_email, detail "
                    "FROM admin_audit ORDER BY ts DESC LIMIT $1",
                    limit,
                )
        out = []
        for r in rows:
            d = r["detail"]
            if isinstance(d, str):
                import json as _json
                try:
                    d = _json.loads(d)
                except Exception:
                    d = {}
            out.append({
                "ts": r["ts"].isoformat() if isinstance(r["ts"], _dt.datetime) else str(r["ts"]),
                "actor_id": r["actor_id"],
                "actor_email": r["actor_email"],
                "action": r["action"],
                "target_id": r["target_id"],
                "target_email": r["target_email"],
                "detail": d or {},
            })
        return out
