# Connecting an AI client to your GML server

GML exposes a long-term memory service through **two parallel surfaces**:

| Surface | Path | Who uses it |
|---|---|---|
| **MCP** (Model Context Protocol) | `https://yourhost/mcp` | Claude Desktop, Claude Code, Cursor, Windsurf, Antigravity |
| **REST `/api/*`** | `https://yourhost/api/*` | Web UIs, ChatGPT Custom GPTs, anything that can speak HTTP |

The live instance is at **`https://akhrots.com`** — all routes are consolidated on that one host (`/mcp`, `/api/*`, `/auth/*`). The examples below use it; for your own deployment, replace `akhrots.com` with your hostname.

> **Note:** point MCP clients **directly at `https://akhrots.com/mcp`**. The old `mcp.akhrots.com` subdomain still works but only 308-redirects here, and HTTP clients drop the `Authorization` header across a cross-host redirect — so a client left on `mcp.akhrots.com` will fail auth.

## Get your API key

You need an API key issued by the server admin. The admin runs:

```bash
curl -X POST https://akhrots.com/api/admin/keys \
  -H "X-API-Key: $MASTER_KEY" \
  -H "content-type: application/json" \
  -d '{"user_id":"<your-user-id>","label":"<machine label>"}'
```

Save the returned `key` — it cannot be retrieved later, only re-issued.

---

## Platform-by-platform

### 1) Claude Desktop (Mac/Windows)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "gml-memory": {
      "url": "https://akhrots.com/mcp",
      "transport": "streamable-http",
      "headers": {
        "X-API-Key": "gml_YOUR_KEY_HERE"
      }
    }
  }
}
```

Quit and restart Claude Desktop. The tools (`query`, `ingest`, `recall`, etc.) appear in the conversation.

### 2) Claude Code (CLI)

```bash
claude mcp add gml-memory \
  --url https://akhrots.com/mcp \
  --transport streamable-http \
  --header "X-API-Key: gml_YOUR_KEY_HERE"
```

Or manually edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "gml-memory": {
      "url": "https://akhrots.com/mcp",
      "transport": "streamable-http",
      "headers": { "X-API-Key": "gml_YOUR_KEY_HERE" }
    }
  }
}
```

### 3) Cursor

Edit `~/.cursor/mcp.json` (or `<project>/.cursor/mcp.json` for project-scoped):

```json
{
  "mcpServers": {
    "gml-memory": {
      "url": "https://akhrots.com/mcp",
      "headers": {
        "X-API-Key": "gml_YOUR_KEY_HERE"
      }
    }
  }
}
```

Reload Cursor (Cmd-Shift-P → "Reload Window"). The MCP panel should list `gml-memory` and its tools.

### 4) Windsurf (Codeium)

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "gml-memory": {
      "serverUrl": "https://akhrots.com/mcp",
      "headers": {
        "X-API-Key": "gml_YOUR_KEY_HERE"
      }
    }
  }
}
```

Restart Windsurf (Cmd-Shift-P → "Developer: Reload Window"). Open the MCP tools panel to confirm `gml-memory` shows up.

> **Note:** Windsurf's HTTP-transport support depends on version. If your build doesn't accept `serverUrl` for a remote server, fall back to running GML locally with `GML_MCP_TRANSPORT=stdio` and pointing Windsurf at the local stdio command (see "Stdio fallback" below).

### 5) Antigravity (Google IDE)

Antigravity uses project-level MCP config. Create `<your-project>/.antigravity/mcp.json`:

```json
{
  "servers": {
    "gml-memory": {
      "url": "https://akhrots.com/mcp",
      "headers": {
        "X-API-Key": "gml_YOUR_KEY_HERE"
      }
    }
  }
}
```

Reload Antigravity. The MCP tools panel surfaces `gml-memory`.

---

## ChatGPT — MCP isn't there yet; use a Custom GPT Action instead

ChatGPT's MCP Connector is in restricted preview as of mid-2026. For a stable, generally-available path, build a **Custom GPT with Actions** pointing at the GML REST `/api/*` surface:

1. Go to **Explore GPTs → Create**.
2. **Configure → Actions → Create new action**.
3. Paste the OpenAPI schema from `docs/openapi-chatgpt.json` (or fetch live from `https://akhrots.com/openapi.json` — both work).
4. **Authentication → API Key**. Type: `Custom`, Header name: `X-API-Key`. Paste your key.
5. **Privacy policy URL** → fill in your real one (ChatGPT requires this for actions).
6. Save. The GPT can now call `/api/memory/recall`, `/api/memory/ingest`, etc.

In the GPT's system prompt, mirror the MCP guidance:

> Before answering any user question, call `recall_memory` with the user's question to fetch any relevant prior context. After answering, call `ingest_memory` with the exchange to persist new facts.

---

## Gemini — no MCP, no good native path

Gemini (consumer app, Workspace, AI Studio) does not implement MCP as of 2026-05. Honest options:

1. **Build a Workspace add-on** that calls the `/api/*` surface. Submitting to the Workspace Marketplace requires Google review; budget ~1–2 weeks for approval.
2. **Vertex AI Agent Builder** can use the `/api/*` OpenAPI spec as a tool, similar to ChatGPT Actions. Limited to Vertex deployments.
3. **Wait** for Google to ship MCP in Gemini Code Assist / Antigravity (the latter is already done — see above). The consumer app appears to be lagging.

None of these are a 5-minute config paste. If Gemini support is a hard requirement, plan a separate sprint.

---

## Stdio fallback (run GML locally, no remote server)

Useful when:
- You're a developer testing changes
- The user's environment can't reach the remote server
- The client doesn't support HTTP transport

```json
{
  "mcpServers": {
    "gml-memory": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "orchestration.mcp_server"],
      "env": {
        "GML_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

This launches a fresh GML process per client; memories persist to `~/.gml/memories.jsonl` on the local machine, NOT the server's.

---

## Verifying the connection

After install, in any MCP-aware client try:

```
Use the gml-memory tool `status` to show the memory layer's current state.
```

A successful `status` response (memory counts, embedder version, etc.) confirms the wiring. If you get "unauthorized", double-check the `X-API-Key` header value.

---

## Tool reference (same across all platforms)

| Tool | Purpose | When to call |
|---|---|---|
| `query(text)` | Full pipeline — classify → retrieve → SAM → assemble. Returns the formatted context the assistant should consume. | **Before** answering every user turn |
| `ingest(user_query, assistant_reply)` | LLM-extracted persistence of facts from the exchange. | **After** answering every user turn |
| `sdp_ingest(user_query, assistant_reply)` | Fast regex-only persistence. 100× faster than `ingest` but catches fewer facts. | When latency matters and the turn is clearly factual |
| `recall(query, top_k)` | Raw retrieval — top-K vector hits, no SAM. | Debug / when the assistant explicitly wants raw memories |
| `remember(content, entity, ...)` | Explicit single-fact save. | When the user says "remember that …" |
| `forget(memory_id)` | Delete one memory. | When the user says "forget that …" |
| `list_memories(entity?, limit?)` | Browse stored memories. | When the user asks "what do you remember about X?" |
| `improve_query(text)` | SAM-rewrite a vague query into a sharper one. | When initial retrieval comes up empty |
| `status()` | Memory store + embedder stats. | Health check |
| `trace(text)` | Run the pipeline and emit every stage's intermediate output. | Deep debugging |
| `analyze(text)` | One-shot pipeline run, returns the formatted response. | Single-shot use without a chat session |
| `diag()` | Diagnostic dump of internal state. | Triage when something's off |
