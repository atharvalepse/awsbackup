# mcp-relay

A multi-user **broker / relay** for the [Model Context Protocol](https://modelcontextprotocol.io).
Both MCP **hosts** (clients) and MCP **servers** dial *into* the relay over HTTP; the
relay pairs them per-user and routes JSON-RPC traffic in both directions. This lets a
browser extension or native app reach an MCP server that runs on someone's machine
behind NAT — neither side needs an inbound port except the relay.

```
 Browser ext ─┐                       ┌─ connector → stdio MCP server (alice)
 Native app  ─┤──▶  mcp-relay  ◀──────┤
 MCP host    ─┘   (HTTP broker)       └─ connector → stdio MCP server (bob)
       host-facing: /mcp              connector-facing: /relay/*
```

## Design

| Concern | Approach |
| --- | --- |
| **Transport** | MCP **Streamable HTTP** on the host side (`/mcp`, POST + SSE) — stock MCP clients work unchanged. The server side dials in via a small **connector**. |
| **Topology** | Broker: servers can't be reached directly, so a connector opens an outbound SSE *downlink* and POSTs replies on an *uplink*. |
| **Multi-tenancy** | A bearer token maps to a `user_id`. A host can only reach servers registered under the **same** user — the token is the trust boundary. |
| **Auth** | Two backends: static `token→user_id` config (dev), or **Postgres** with email+password signup/login and per-user API tokens (issued, hashed, revocable) — `/auth/*` + a `/dashboard`. |
| **Routing** | Each host session gets its **own logical session** on the backing server (one connector subprocess per session), so JSON-RPC ids never collide and never need rewriting — correlation is exact. |
| **Scale** | Single process, in-memory tables, asyncio. (Swap in Redis pub/sub later for horizontal scale.) |
| **Browsers** | CORS is enabled and `?token=`/`?session=` query fallbacks exist because `EventSource` can't set headers. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the demo

1. **Config** — copy the example and pick tokens:

   ```bash
   cp relay_config.example.json relay_config.json
   ```

2. **Start the relay:**

   ```bash
   mcp-relay --host 127.0.0.1 --port 8080
   # or: python -m mcp_relay
   ```

3. **Connect a server** (bridges the bundled stdio echo server in as user `alice`):

   ```bash
   mcp-relay-connector --url http://127.0.0.1:8080 \
       --token alice-dev-token-change-me --name echo \
       -- python examples/echo_server.py
   ```

   Or the **real filesystem server**, sandboxed to a directory:

   ```bash
   mcp-relay-connector --url http://127.0.0.1:8080 \
       --token alice-dev-token-change-me --name filesystem \
       -- python examples/filesystem_server.py ~/mcp-sandbox
   ```

4. **Act as a host:**

   ```bash
   python examples/demo_client.py --url http://127.0.0.1:8080 \
       --token alice-dev-token-change-me --server echo --text "hello relay"
   ```

   ```
   initialize -> {"name": "echo-server", "version": "0.1.0"}
   tools/list -> ['echo']
   tools/call -> echo: hello relay
   ```

## Use it from an IDE

Most IDEs launch an MCP server as a stdio command, so the relay ships
`mcp-relay-client` — a bridge that proxies stdio ⇄ the relay's `/mcp`. Paste this
into your IDE's MCP config (full per-IDE guide in [`docs/ide-setup.md`](docs/ide-setup.md)):

```json
{
  "mcpServers": {
    "relay-all":  { "command": "mcp-relay-client",
                    "args": ["--url", "https://relay.example.com", "--all"],
                    "env": { "RELAY_TOKEN": "alice-dev-token-change-me" } },
    "relay-echo": { "command": "mcp-relay-client",
                    "args": ["--url", "https://relay.example.com", "--server", "echo"],
                    "env": { "RELAY_TOKEN": "alice-dev-token-change-me" } }
  }
}
```

- `--server NAME` exposes one backend (a **separate** MCP per server).
- `--all` exposes the **aggregated** view of every server you have (`server=*`):
  one MCP entry, tools namespaced `<server>__<tool>`, calls routed to the owner.
- IDEs with native remote-MCP support can skip the bridge and point straight at
  `https://relay.example.com/mcp?server=*` with an `Authorization: Bearer` header.

## Authentication & tokens

Two backends, chosen at startup:

**Static (dev)** — `relay_config.json` maps `token → user_id`. Zero infra; fine for
local use. This is the default when no database is configured.

**Postgres (real users)** — pass `--database-url` (or `DATABASE_URL`) and set
`RELAY_SESSION_SECRET`:

```bash
createdb mcp_relay
DATABASE_URL=postgresql:///mcp_relay RELAY_SESSION_SECRET=$(openssl rand -hex 32) \
  mcp-relay --port 8080
```

This enables email+password accounts and per-user API tokens. **How a token is
generated and shared** (this is the credential you paste into an IDE):

1. User signs up / logs in at **`/dashboard`** (or via the API below) → gets a
   short-lived login **session**.
2. With that session they **create an API token** → the relay returns a random
   token **once** (only its SHA-256 is stored) and they copy it.
3. They paste that token into their IDE config as `RELAY_TOKEN` — done.
4. Tokens can be listed and **revoked** from the dashboard; revocation takes
   effect within the token cache TTL (~15s).

| Method | Path | Body / auth | Returns |
| --- | --- | --- | --- |
| `POST` | `/auth/signup` | `{email, password}` | `{user_id, email, session}` |
| `POST` | `/auth/login` | `{email, password}` | `{user_id, session}` |
| `POST` | `/auth/tokens` | `Bearer <session>` + `{label}` | `{id, token, label}` (token shown once) |
| `GET` | `/auth/tokens` | `Bearer <session>` | `{tokens: [...]}` (no raw values) |
| `DELETE` | `/auth/tokens/{id}` | `Bearer <session>` | `204` |
| `GET` | `/dashboard` | — | minimal signup/login/token HTML UI |

Passwords are PBKDF2-HMAC-SHA256 (stdlib). Login sessions are HMAC-signed bearer
strings. The same `user_id` is the routing/isolation key the relay already uses,
so DB users get the same per-user server isolation as static tokens.

## HTTP API

### Host-facing (standard MCP Streamable HTTP)

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/mcp` | Send JSON-RPC. Requests stream their reply back as `text/event-stream` and the stream closes; notifications/responses return `202`. First call must be `initialize` (no session yet); the response carries `Mcp-Session-Id`. Pick a server with `?server=<name>` (optional if the user owns exactly one), or `?server=*` for the aggregated view of all your servers. |
| `GET` | `/mcp` | Open the standby SSE stream for **server-initiated** messages (sampling, notifications). Requires `Mcp-Session-Id`. |
| `DELETE` | `/mcp` | Terminate the session. |

Auth: `Authorization: Bearer <token>` (or `?token=` for `EventSource`).

### Connector-facing

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/relay/register` | `{ "server": "<name>", "info": {...} }` → `{ "server_session": "..." }`. Auth with the user bearer token. |
| `GET` | `/relay/stream` | Downlink SSE of envelopes (`open` / `message` / `close`). Auth with `X-Relay-Server-Session`. |
| `POST` | `/relay/message` | Uplink: `{ "session": "<id>", "payload": <json-rpc> }`. |

### Meta

`GET /health` → status + live stats. `GET /` → human-readable info.

## Routing internals

* **`open`** — relay tells the connector a new host session started; the connector spawns a dedicated subprocess.
* **`message`** — host→server JSON-RPC (requests, notifications, and responses to server-initiated requests) all travel on the downlink.
* Server→host messages arrive on the uplink and are routed: a **response** to an outstanding host request resolves that request's pending future (delivered on the originating POST's SSE stream); everything else lands on the host's **standby** GET stream.
* **`close`** — host session ended; the connector tears the subprocess down.

### Aggregation (`server=*`)

`aggregator.py` layers a multiplexer on top of the router (the pure 1:1 path is
untouched). One host session spans N backend sub-sessions:

* `initialize` fans out; capabilities are unioned, `serverInfo` is synthetic.
* `tools/list` / `prompts/list` / `resources/list` are fanned out and merged with
  names namespaced `<server>__<name>`; the relay records each name's owner.
* `tools/call` / `prompts/get` / `resources/read` are routed to the owning backend.
* Fan-out ids are rewritten (`agg:N`) for merge correlation; server-initiated ids
  are rewritten (`si:N`) so the host's reply routes back to the right backend.
  Single-target calls keep the host id (only one backend sees it).

## Tests

```bash
pytest
```

The suite boots a real uvicorn server in a thread and exercises initialize/list/call,
multi-user isolation, auth failures, default-server selection, notifications,
**aggregation across two servers** (namespacing + routing), and a full end-to-end run
through the actual `mcp-relay-client` stdio bridge subprocess.

## Limitations / next steps

* In-memory and single-process. Restarting the relay drops all sessions (hosts re-`initialize`). Add Redis pub/sub + a shared session store to scale out.
* Auth has email+password + API tokens (Postgres). Still missing: email verification, password reset, rate limiting, and **MCP OAuth** (so IDEs can "Connect" without pasting a token).
* SSE stream resumption (`Last-Event-ID`) is not implemented.
* If a connector process restarts it re-registers with a fresh session; previously-attached hosts should re-`initialize`.
