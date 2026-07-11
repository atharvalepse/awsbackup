"""Loading the user/token table.

Resolution order:
  1. explicit ``path`` argument
  2. ``RELAY_CONFIG`` env var (path to a JSON file)
  3. ``./relay_config.json`` if present
  4. ``RELAY_TOKENS`` env var: "user1:token1,user2:token2"

The result is a mapping of ``token -> user_id``. Tokens are the trust boundary:
anyone holding a user's token can register servers as that user and reach that
user's servers, and nobody else's.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class ConfigError(Exception):
    pass


def _from_file(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    tokens: dict[str, str] = {}
    for entry in data.get("users", []):
        uid = entry.get("id")
        token = entry.get("token")
        if not uid or not token:
            raise ConfigError(f"user entry missing id/token: {entry!r}")
        if token in tokens:
            raise ConfigError(f"duplicate token for users {tokens[token]!r} and {uid!r}")
        tokens[token] = uid
    return tokens


def _from_env_tokens(spec: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ConfigError(f"RELAY_TOKENS entry must be 'user:token', got {pair!r}")
        uid, token = pair.split(":", 1)
        tokens[token.strip()] = uid.strip()
    return tokens


def load_tokens(path: str | None = None) -> dict[str, str]:
    if path:
        return _from_file(Path(path))

    env_path = os.environ.get("RELAY_CONFIG")
    if env_path:
        return _from_file(Path(env_path))

    default = Path("relay_config.json")
    if default.exists():
        return _from_file(default)

    env_tokens = os.environ.get("RELAY_TOKENS")
    if env_tokens:
        return _from_env_tokens(env_tokens)

    raise ConfigError(
        "No configuration found. Provide --config, set RELAY_CONFIG or RELAY_TOKENS, "
        "or create relay_config.json (see relay_config.example.json)."
    )
