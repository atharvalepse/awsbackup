"""Password hashing — PBKDF2-HMAC-SHA256 (stdlib only, no native dep).

Self-describing hash format (Django-compatible layout):

    pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>

We deliberately avoid bcrypt/argon2 to keep the install dependency-free (the
project ships no compiled crypto for this). PBKDF2-HMAC-SHA256 at a high
iteration count is a sound, FIPS-friendly password KDF. Bump ``ITERATIONS``
over time; verify() reads the count from the stored string, so old hashes keep
verifying and you can transparently re-hash on next login.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

ALGO = "pbkdf2_sha256"
ITERATIONS = 600_000          # OWASP-recommended floor for PBKDF2-SHA256 (2023+)
SALT_BYTES = 16


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def hash_password(password: str, *, iterations: int = ITERATIONS) -> str:
    """Return a self-describing ``pbkdf2_sha256$iters$salt$hash`` string."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGO}${iterations}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time check of ``password`` against a stored hash string."""
    if not stored or not password:
        return False
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$")
        if algo != ALGO:
            return False
        iterations = int(iters_s)
        salt = _unb64(salt_b64)
        expected = _unb64(hash_b64)
    except (ValueError, base64.binascii.Error):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)
