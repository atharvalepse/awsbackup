"""Authentication: users, passwords, login sessions, and API tokens.

Two backends implement the same ``user_for_token`` contract the relay needs:

* ``StaticAuth`` — the original ``token -> user_id`` dict from config (dev/tests).
* ``DatabaseAuth`` — Postgres-backed: email+password users, hashed API tokens,
  issuance/revocation, with an in-memory cache on the hot ``user_for_token`` path.

Passwords use PBKDF2-HMAC-SHA256 (stdlib — no native build deps). API tokens are
high-entropy random strings; only their SHA-256 is stored. Login sessions are
short-lived HMAC-signed bearer strings (a tiny JWT-alike) used to manage tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Optional

try:  # only needed for the Postgres backend
    import psycopg
    from psycopg_pool import AsyncConnectionPool
except ImportError:  # pragma: no cover - static mode doesn't need psycopg
    psycopg = None
    AsyncConnectionPool = None

PBKDF2_ITERATIONS = 600_000
TOKEN_CACHE_TTL = 15.0  # seconds


class EmailTaken(Exception):
    pass


# -- base64 (url-safe, unpadded) -------------------------------------------
def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# -- passwords --------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), _b64d(salt_b64), int(iters))
        return hmac.compare_digest(dk, _b64d(hash_b64))
    except (ValueError, TypeError):
        return False


# A constant to compare against for unknown emails, to blunt timing-based
# account enumeration (we still do the same PBKDF2 work either way).
_DUMMY_HASH = hash_password("dummy-password-for-constant-time")


# -- login sessions (HMAC-signed) ------------------------------------------
def make_session(user_id: str, secret: str, ttl: int = 86_400) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + ttl}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_session(token: str, secret: str) -> Optional[str]:
    try:
        body, sig = token.split(".")
        expected = _b64e(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64d(body))
        if int(payload["exp"]) < time.time():
            return None
        return str(payload["sub"])
    except (ValueError, KeyError, TypeError):
        return None


def _token_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# -- static backend ---------------------------------------------------------
class StaticAuth:
    """The legacy ``token -> user_id`` table from config."""

    def __init__(self, tokens: dict[str, str]):
        self._tokens = dict(tokens)

    async def user_for_token(self, token: Optional[str]) -> Optional[str]:
        return self._tokens.get(token) if token else None

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


# -- Postgres backend -------------------------------------------------------
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id            BIGSERIAL PRIMARY KEY,
        email         TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    """CREATE TABLE IF NOT EXISTS api_tokens (
        id           BIGSERIAL PRIMARY KEY,
        user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash   TEXT UNIQUE NOT NULL,
        label        TEXT,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_used_at TIMESTAMPTZ,
        revoked_at   TIMESTAMPTZ
    )""",
    "CREATE INDEX IF NOT EXISTS idx_api_tokens_user ON api_tokens(user_id)",
]


class DatabaseAuth:
    def __init__(self, dsn: str, session_secret: str, *, min_size: int = 1, max_size: int = 10):
        if AsyncConnectionPool is None:
            raise RuntimeError("psycopg is not installed; install mcp-relay[postgres]")
        self.session_secret = session_secret
        self.pool = AsyncConnectionPool(dsn, open=False, min_size=min_size, max_size=max_size)
        self._cache: dict[str, tuple[Optional[str], float]] = {}

    async def startup(self) -> None:
        await self.pool.open()
        await self.init_schema()

    async def init_schema(self) -> None:
        async with self.pool.connection() as conn, conn.cursor() as cur:
            for stmt in _SCHEMA:
                await cur.execute(stmt)

    async def shutdown(self) -> None:
        await self.pool.close()

    # -- users ----------------------------------------------------------
    async def create_user(self, email: str, password: str) -> str:
        email = email.strip().lower()
        ph = hash_password(password)
        async with self.pool.connection() as conn, conn.cursor() as cur:
            try:
                await cur.execute(
                    "INSERT INTO users(email, password_hash) VALUES (%s, %s) RETURNING id",
                    (email, ph),
                )
            except psycopg.errors.UniqueViolation:
                raise EmailTaken(email)
            row = await cur.fetchone()
        return str(row[0])

    async def verify_login(self, email: str, password: str) -> Optional[str]:
        email = email.strip().lower()
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
            row = await cur.fetchone()
        if row is None:
            verify_password(password, _DUMMY_HASH)  # constant-ish time
            return None
        return str(row[0]) if verify_password(password, row[1]) else None

    # -- tokens ---------------------------------------------------------
    async def create_token(self, user_id: str, label: Optional[str] = None) -> dict:
        raw = secrets.token_urlsafe(32)
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO api_tokens(user_id, token_hash, label) VALUES (%s, %s, %s) "
                "RETURNING id, created_at",
                (int(user_id), _token_hash(raw), label),
            )
            row = await cur.fetchone()
        return {"id": row[0], "token": raw, "label": label, "created_at": row[1].isoformat()}

    async def list_tokens(self, user_id: str) -> list[dict]:
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT id, label, created_at, last_used_at, revoked_at "
                "FROM api_tokens WHERE user_id = %s ORDER BY id",
                (int(user_id),),
            )
            rows = await cur.fetchall()
        return [
            {"id": r[0], "label": r[1],
             "created_at": r[2].isoformat() if r[2] else None,
             "last_used_at": r[3].isoformat() if r[3] else None,
             "revoked": r[4] is not None}
            for r in rows
        ]

    async def revoke_token(self, user_id: str, token_id: int) -> bool:
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE api_tokens SET revoked_at = now() "
                "WHERE id = %s AND user_id = %s AND revoked_at IS NULL RETURNING token_hash",
                (int(token_id), int(user_id)),
            )
            row = await cur.fetchone()
        if row is None:
            return False
        self._cache.pop(row[0], None)  # evict so the relay stops honouring it now
        return True

    # -- hot path -------------------------------------------------------
    async def user_for_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        th = _token_hash(token)
        now = time.time()
        cached = self._cache.get(th)
        if cached is not None and cached[1] > now:
            return cached[0]
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE api_tokens SET last_used_at = now() "
                "WHERE token_hash = %s AND revoked_at IS NULL RETURNING user_id",
                (th,),
            )
            row = await cur.fetchone()
        uid = str(row[0]) if row else None
        self._cache[th] = (uid, now + TOKEN_CACHE_TTL)
        return uid


# -- GML-backed backend (unified auth) -------------------------------------
class GmlAuth:
    """The single source of truth: GML accounts.

    A bearer token is a *GML* credential — either a GML access JWT (HS256,
    ``GML_JWT_SECRET``) or a GML API key (``user_keys``). It resolves to the gml
    ``user_id``, which the relay forwards as the MCP tenant. There are no
    separate relay accounts: the same account you sign into on the web reaches
    the same per-user memory over MCP. This reuses ``orchestration.auth`` and
    the gml user-key store directly (we live in the same repo now).
    """

    def __init__(self) -> None:
        self._store = None  # gml user-key store, for API-key lookup

    async def startup(self) -> None:
        # API-key lookup needs the gml user store (Postgres in prod). JWTs work
        # without it; if it can't be built we degrade to JWT-only rather than fail.
        try:
            from orchestration.storage import make_user_key_store

            self._store = await make_user_key_store()
        except Exception:
            self._store = None

    async def shutdown(self) -> None:
        store, self._store = self._store, None
        closer = getattr(store, "aclose", None) or getattr(store, "close", None)
        if closer is not None:
            try:
                result = closer()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass

    async def user_for_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        from orchestration.auth.tokens import (
            JWTError,
            decode_access_token,
            looks_like_jwt,
        )

        if looks_like_jwt(token):
            try:
                return decode_access_token(token)
            except JWTError:
                pass
        if self._store is not None:
            try:
                rec = await self._store.lookup(token)
            except Exception:
                rec = None
            return rec.user_id if rec else None
        return None
