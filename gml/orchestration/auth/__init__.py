"""Email + password authentication: password hashing + JWT session tokens.

Layered on top of the existing API-key auth (orchestration.users /
PostgresUserKeyStore). A request authenticates via the master key, a per-user
API key, OR a JWT issued by /auth/login — all resolve to a ``user_id`` that
the RLS policies scope memories to.
"""
from orchestration.auth.passwords import hash_password, verify_password
from orchestration.auth.tokens import (
    JWTError,
    decode_access_token,
    make_access_token,
)

__all__ = [
    "hash_password",
    "verify_password",
    "make_access_token",
    "decode_access_token",
    "JWTError",
]
