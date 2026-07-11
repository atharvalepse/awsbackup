"""JWT session tokens (HS256, PyJWT).

The signing secret comes from ``GML_JWT_SECRET``. If unset we fall back to
``GML_API_KEY`` (so a single-secret deploy still works) and, failing that, a
process-random secret — which means tokens won't survive a restart and every
process has its own, so production MUST set ``GML_JWT_SECRET`` (we warn once).
"""
from __future__ import annotations

import os
import secrets
import time

import jwt  # PyJWT

from orchestration.observability.logging import StructuredLogger

slog = StructuredLogger("auth.tokens")

ALGO = "HS256"
DEFAULT_TTL_SECONDS = int(os.environ.get("GML_JWT_TTL_SECONDS", str(7 * 24 * 3600)))

# Resolve the secret once at import. A process-random fallback is fine for dev
# but useless across restarts/replicas — warn so prod sets a real one.
_secret = (
    os.environ.get("GML_JWT_SECRET", "").strip()
    or os.environ.get("GML_API_KEY", "").strip()
)
if not _secret:
    _secret = secrets.token_hex(32)
    slog.warning(
        event="jwt_secret_unset",
        note="GML_JWT_SECRET and GML_API_KEY unset — using a process-random "
             "JWT secret; tokens won't survive a restart. Set GML_JWT_SECRET "
             "in production.",
    )


class JWTError(Exception):
    """Token was missing, malformed, expired, or signed with the wrong key."""


def make_access_token(user_id: str, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> dict:
    """Mint an access token for ``user_id``. Returns the OAuth-style envelope."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + ttl_seconds,
        "type": "access",
    }
    token = jwt.encode(payload, _secret, algorithm=ALGO)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ttl_seconds,
        "user_id": user_id,
    }


def decode_access_token(token: str) -> str:
    """Verify ``token`` and return its ``user_id`` (sub). Raises JWTError."""
    if not token:
        raise JWTError("empty token")
    try:
        payload = jwt.decode(token, _secret, algorithms=[ALGO])
    except jwt.PyJWTError as exc:
        raise JWTError(str(exc)) from exc
    sub = payload.get("sub")
    if not sub or payload.get("type") != "access":
        raise JWTError("missing sub / wrong token type")
    return sub


def looks_like_jwt(token: str) -> bool:
    """Cheap shape check: a JWT is three base64url segments split by dots.

    Lets the auth middleware route a bearer credential to JWT-verify vs.
    API-key lookup without a DB round-trip on every request."""
    return token.count(".") == 2 and " " not in token
