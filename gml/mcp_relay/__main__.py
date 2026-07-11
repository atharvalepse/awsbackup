"""Run the relay: ``python -m mcp_relay`` or the ``mcp-relay`` console script.

Auth modes:
* ``GML_JWT_SECRET`` set (or ``--gml-auth``) → unified GML account auth: a
  bearer is a GML access JWT or GML API key, resolved to the gml user_id (the
  MCP tenant). One account across web + MCP; no relay-native accounts/dashboard.
* otherwise → the static ``token -> user_id`` config table (``--config`` /
  ``RELAY_CONFIG`` / ``RELAY_TOKENS`` / ``relay_config.json``) for dev/tests.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys

from .config import ConfigError, load_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcp-relay", description="MCP broker/relay")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="bind port (default 8080)")
    parser.add_argument("--config", help="path to static JSON token config")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"),
                        help="(legacy) Postgres DSN for the old relay-native auth; unused with GML auth")
    parser.add_argument("--gml-auth", action="store_true",
                        help="force unified GML account auth (JWT / API key); implied when GML_JWT_SECRET is set")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    from .app import create_app
    from .registry import RelayState

    # Unified auth: GML accounts are the single source of truth. When a GML JWT
    # secret is configured (production), validate bearers as GML credentials
    # (a GML access JWT or a GML API key) and forward the resolved gml user_id
    # as the MCP tenant — there are no separate relay accounts or /dashboard.
    # Otherwise fall back to the static token table for local dev/tests.
    if os.environ.get("GML_JWT_SECRET") or args.gml_auth:
        from .auth import GmlAuth

        app = create_app(RelayState(GmlAuth()))
        print(f"mcp-relay on http://{args.host}:{args.port}  "
              f"(auth: GML accounts — JWT / API key)")
    else:
        try:
            tokens = load_tokens(args.config)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        app = create_app(RelayState(tokens))
        n_users = len(set(tokens.values()))
        print(f"mcp-relay on http://{args.host}:{args.port}  "
              f"(auth: static tokens, {n_users} user(s))")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
