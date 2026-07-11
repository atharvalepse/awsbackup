# Connectors

Server-side helpers that bake a **per-user MCP key** into the install artifacts
the `/api/me/install/*` surface serves (see the "personalized MCP install
surface" in [`orchestration/api_routes.py`](../api_routes.py)).

## `codex_bridge.js` — vendored, do not hand-edit

A verbatim copy of `bridge/index.js` from
[Sakshxm-py/akhrot-codex-mcp](https://github.com/Sakshxm-py/akhrot-codex-mcp)
(MIT — see [`CODEX_BRIDGE_LICENSE`](./CODEX_BRIDGE_LICENSE)).

OpenAI Codex (and Claude Desktop) speak MCP over **stdio**, while GML exposes a
remote **streamable-HTTP** endpoint at `/mcp`. This zero-dependency Node bridge
proxies stdio ⇄ HTTPS, authenticating with `Authorization: Bearer $GML_TOKEN` —
the same per-user `gml_…` key `PostgresUserKeyStore.issue()` mints for every
other client. It also rewrites the tool list/descriptions so the memory tools
fire only on the trigger phrase ("use akhrots memory"); that gating is the
reason Codex goes through the bridge instead of a plain HTTP connector.

The file is kept **byte-identical** to upstream so its checksum can be verified
against the reviewed source. To update, re-copy from upstream rather than
editing in place. The token is never baked into this file — it is injected at
download time via the `env` block of the generated `config.toml` / `.mcp.json`
(see [`codex.py`](./codex.py)).
