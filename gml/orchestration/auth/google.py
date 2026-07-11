"""Google Identity Services ID-token verification.

The web login page renders Google's "Continue with Google" button (Google
Identity Services). On success the browser hands us a *credential* — a signed
ID token (JWT) issued by Google. We verify it here, server-side:

  * signature is checked against Google's published certs (cached by the
    google-auth library), so a forged token is rejected;
  * the ``aud`` claim must equal our own OAuth client id (``GML_GOOGLE_CLIENT_ID``)
    — a token minted for some *other* site won't authenticate here;
  * the ``iss`` is one of Google's issuers (google-auth enforces this).

We never see (or need) the Google client *secret* for this flow — ID-token
verification is public-key only.
"""
from __future__ import annotations

import os

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from orchestration.observability.logging import StructuredLogger

slog = StructuredLogger("auth.google")


class GoogleAuthError(Exception):
    """The Google credential was missing, malformed, expired, untrusted, or
    minted for a different client id / unverified email."""


# One transport is enough; google-auth caches Google's signing certs on it.
_transport = google_requests.Request()


def _client_id() -> str:
    cid = (os.environ.get("GML_GOOGLE_CLIENT_ID", "").strip())
    if not cid:
        raise GoogleAuthError("Google sign-in is not configured on this server")
    return cid


def verify_google_credential(credential: str) -> dict:
    """Verify a Google ID token and return the trusted claims we care about.

    Returns ``{"sub", "email", "email_verified", "name"}``. Raises
    :class:`GoogleAuthError` on any verification failure.
    """
    if not credential:
        raise GoogleAuthError("missing Google credential")
    client_id = _client_id()
    try:
        claims = google_id_token.verify_oauth2_token(
            credential, _transport, client_id
        )
    except ValueError as exc:
        # verify_oauth2_token raises ValueError for bad signature, wrong
        # audience, expiry, malformed token, etc.
        raise GoogleAuthError(f"invalid Google credential: {exc}") from exc

    email = (claims.get("email") or "").strip().lower()
    if not email:
        raise GoogleAuthError("Google account has no email")
    # email_verified comes back as a bool or the string "true".
    verified = claims.get("email_verified")
    if verified not in (True, "true", "True"):
        raise GoogleAuthError("Google email is not verified")

    return {
        "sub": claims.get("sub"),
        "email": email,
        "email_verified": True,
        "name": claims.get("name"),
    }
