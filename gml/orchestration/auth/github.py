"""GitHub OAuth (authorization-code flow).

Unlike Google (which hands us a verifiable ID token in the browser), GitHub
uses the classic server-side code exchange:

  1. browser is sent to github.com/login/oauth/authorize
  2. GitHub redirects back to /auth/github/callback?code=...&state=...
  3. we exchange the code for an access token using the client secret
  4. we read the user's *primary, verified* email from the GitHub API

The client id/secret come from ``GML_GITHUB_CLIENT_ID`` /
``GML_GITHUB_CLIENT_SECRET``.
"""
from __future__ import annotations

import os

import httpx

from orchestration.observability.logging import StructuredLogger

slog = StructuredLogger("auth.github")

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
API_BASE = "https://api.github.com"


class GitHubAuthError(Exception):
    """OAuth handshake failed, or no usable verified email on the account."""


def github_config() -> tuple[str, str]:
    cid = os.environ.get("GML_GITHUB_CLIENT_ID", "").strip()
    secret = os.environ.get("GML_GITHUB_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise GitHubAuthError("GitHub sign-in is not configured on this server")
    return cid, secret


def authorize_url(state: str, redirect_uri: str) -> str:
    cid, _ = github_config()
    from urllib.parse import urlencode
    q = urlencode({
        "client_id": cid,
        "redirect_uri": redirect_uri,
        "scope": "read:user user:email",
        "state": state,
        "allow_signup": "true",
    })
    return f"{AUTHORIZE_URL}?{q}"


async def exchange_code(code: str, redirect_uri: str) -> str:
    """Trade the one-time ``code`` for a GitHub access token."""
    cid, secret = github_config()
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": cid,
                "client_secret": secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
    if r.status_code != 200:
        raise GitHubAuthError(f"token exchange failed (HTTP {r.status_code})")
    body = r.json()
    tok = body.get("access_token")
    if not tok:
        raise GitHubAuthError(body.get("error_description") or "no access token returned")
    return tok


async def fetch_identity(access_token: str) -> dict:
    """Return ``{email, login, id, name}`` for the token holder. Uses the
    *primary, verified* email; raises if there isn't one."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "akhrot-auth",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        ur = await client.get(f"{API_BASE}/user", headers=headers)
        er = await client.get(f"{API_BASE}/user/emails", headers=headers)
    if ur.status_code != 200:
        raise GitHubAuthError(f"GitHub /user failed (HTTP {ur.status_code})")
    user = ur.json()
    email = None
    if er.status_code == 200:
        emails = er.json()
        primary = next(
            (e for e in emails if e.get("primary") and e.get("verified")), None
        )
        any_verified = next((e for e in emails if e.get("verified")), None)
        chosen = primary or any_verified
        if chosen:
            email = (chosen.get("email") or "").strip().lower()
    if not email:
        # Fall back to the public profile email only if it's present (it is
        # not guaranteed verified, so prefer the /user/emails path above).
        raise GitHubAuthError(
            "no verified email on your GitHub account — add & verify one, "
            "or grant the email permission"
        )
    return {
        "email": email,
        "login": user.get("login"),
        "id": str(user.get("id")) if user.get("id") is not None else None,
        "name": user.get("name"),
    }
