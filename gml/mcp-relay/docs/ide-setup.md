# Using the relay from IDEs & agents

Most MCP-capable tools (Claude, Cursor, Codex, Gemini, Windsurf, VS Code, …)
launch an MCP server as a **stdio command**. `mcp-relay-client` is that command:
it bridges the tool's stdio MCP transport to a remote relay over Streamable HTTP,
and opens the standby SSE stream so server-initiated messages reach the tool.

```
Tool/agent ──stdio──▶ mcp-relay-client ──HTTP /mcp?server=*──▶ relay ──▶ your MCP servers
```

**The whole trick:** there is *one* bridge binary. Every tool below runs the same
`mcp-relay-client --all` (or `--server NAME`); only the config file's *location*
and *dialect* differ.

## Install the bridge

```bash
pipx install mcp-relay        # or: uv tool install mcp-relay
# or, from this repo:
pip install -e .
```

This puts `mcp-relay-client` on your PATH. Confirm: `which mcp-relay-client`.

> ⚠️ **PATH gotcha:** GUI apps (Claude Desktop, Cursor, Windsurf, Antigravity)
> often don't inherit your shell `PATH`, so `"command": "mcp-relay-client"` may
> fail with *command not found*. Use the **absolute path** (`pipx` →
> `~/.local/bin/mcp-relay-client`) or `uvx` (below).

## Target selection

| Flag | Reaches |
| --- | --- |
| `--all` | every server you have, **aggregated** into one MCP (`server=*`); tools namespaced `<server>__<name>` |
| `--server NAME` | exactly that one server (a **separate** MCP per backend) |
| *(neither)* | your single server, if you have exactly one |

The token (`RELAY_TOKEN`) is per user — it scopes you to *your* servers only.

## How it works (example: Cursor)

1. You save `~/.cursor/mcp.json` (below). Cursor launches the `command` as a
   long-lived child process and talks to it over stdio — it thinks it's a normal
   local MCP server.
2. The bridge authenticates with `Bearer $RELAY_TOKEN`, opens a session against
   `?server=*`, and the relay fans out to every server you've connected.
3. Cursor sees **one** server, `relay-all`, exposing tools named
   `<server>__<tool>` (e.g. `filesystem__read_file`). A call goes
   bridge → relay → owning backend → back up the same stdio pipe.

```
Cursor
  └─ mcp-relay-client --all        (stdio child, env RELAY_TOKEN=...)
        └─ HTTPS ─▶ relay ─▶ connectors ─▶ your MCP servers
```

---

## Per-tool configs

Replace `https://relay.example.com` and the token with your own.

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) /
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "relay-all": {
      "command": "mcp-relay-client",
      "args": ["--url", "https://relay.example.com", "--all"],
      "env": { "RELAY_TOKEN": "alice-dev-token-change-me" }
    },
    "relay-echo": {
      "command": "mcp-relay-client",
      "args": ["--url", "https://relay.example.com", "--server", "echo"],
      "env": { "RELAY_TOKEN": "alice-dev-token-change-me" }
    }
  }
}
```

Keep both the aggregated entry and per-server entries — they coexist.

### Claude Code (CLI)

```bash
claude mcp add relay-all --env RELAY_TOKEN=alice-dev-token-change-me \
  -- mcp-relay-client --url https://relay.example.com --all
```

(or add a project-scoped `.mcp.json` with the same `mcpServers` block as Claude Desktop.)

### Cursor

`~/.cursor/mcp.json` (global) or `<project>/.cursor/mcp.json` — same `mcpServers`
shape as Claude Desktop.

### Windsurf

`~/.codeium/windsurf/mcp_config.json` (or **Settings → Cascade → MCP → Add
server**) — same `mcpServers` shape.

### VS Code (GitHub Copilot)

`.vscode/mcp.json` — note the key is **`servers`** and each entry names a `type`:

```json
{
  "servers": {
    "relay-all": {
      "type": "stdio",
      "command": "mcp-relay-client",
      "args": ["--url", "https://relay.example.com", "--all"],
      "env": { "RELAY_TOKEN": "alice-dev-token-change-me" }
    }
  }
}
```

### Codex CLI (OpenAI)

`~/.codex/config.toml` — **TOML**, table key `mcp_servers`:

```toml
[mcp_servers.relay-all]
command = "mcp-relay-client"
args = ["--url", "https://relay.example.com", "--all"]
env = { RELAY_TOKEN = "alice-dev-token-change-me" }
```

### Gemini CLI

`~/.gemini/settings.json` (or project `.gemini/settings.json`):

```json
{
  "mcpServers": {
    "relay-all": {
      "command": "mcp-relay-client",
      "args": ["--url", "https://relay.example.com", "--all"],
      "env": { "RELAY_TOKEN": "alice-dev-token-change-me" }
    }
  }
}
```

### Zed

`settings.json` → key `context_servers`:

```json
{
  "context_servers": {
    "relay-all": {
      "command": { "path": "mcp-relay-client",
                   "args": ["--url", "https://relay.example.com", "--all"],
                   "env": { "RELAY_TOKEN": "alice-dev-token-change-me" } }
    }
  }
}
```

### Antigravity (Google)

> ⚠️ Antigravity takes the standard stdio `mcpServers` JSON, but add it through
> its in-app **MCP / Agent settings** panel — verify the exact file path in-app
> (it changes across releases). The block is identical to Cursor's.

### Perplexity

> ⚠️ Perplexity's MCP support is **remote connectors only** — it does not run a
> local stdio command, so it can't use the bridge. Point it at the relay
> directly via its Connectors UI (see *Native remote MCP* below).

---

## Without installing (uvx / pipx run)

```json
{
  "command": "uvx",
  "args": ["--from", "mcp-relay", "mcp-relay-client",
           "--url", "https://relay.example.com", "--all"],
  "env": { "RELAY_TOKEN": "alice-dev-token-change-me" }
}
```

## Native remote MCP (no bridge)

The relay's host endpoint is standard MCP Streamable HTTP, so any client with
native remote-MCP support can point straight at it — including the aggregated view:

```
URL:    https://relay.example.com/mcp?server=*
Header: Authorization: Bearer alice-dev-token-change-me
```

Use a single server name instead of `*` to target one backend. If a client can't
set custom headers, the relay also accepts the token in the query (less secure —
it lands in logs):

```
https://relay.example.com/mcp?server=*&token=alice-dev-token-change-me
```

## Troubleshooting

* **command not found** — PATH gotcha above; use an absolute path or `uvx`.
* **server unavailable / empty tool list** — a connector for that server must be
  running; check `curl https://relay.example.com/health` (`servers_connected`).
* **401** — wrong/missing `RELAY_TOKEN`.
* **tools missing under `--all`** — names are namespaced `<server>__<tool>`.
