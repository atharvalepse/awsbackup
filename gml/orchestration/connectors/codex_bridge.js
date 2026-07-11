#!/usr/bin/env node
/**
 * Akhrot GML Memory - self-contained stdio <-> streamable-HTTP proxy.
 *
 * Codex (and Claude Desktop) speak MCP over stdio (newline-delimited JSON-RPC).
 * The GML server is a remote streamable-HTTP endpoint that authenticates with a
 * bearer token. This bridge forwards each stdio message to the endpoint via
 * POST, returns the JSON/SSE response back over stdout, and (best-effort) opens
 * a GET SSE channel for server-initiated messages. No OAuth, no npx, no deps.
 *
 * Token handling: GML_TOKEN is injected by the generator at download time and
 * passed in as an env var by Codex (config.toml `env` / plugin `.mcp.json`).
 * It is a short-TTL per-user token - rotate by re-downloading the installer.
 * This file NEVER contains a baked token of its own.
 */
"use strict";

const URL_ = (process.env.GML_MCP_URL || "https://akhrots.com/mcp").trim();
const TOKEN = (process.env.GML_TOKEN || "").trim();

// Best-effort log file next to this script, so we can see whether the host
// (Codex) actually launched the bridge and what happened. Never throws.
let LOG_FILE = null;
try {
  const p = require("path");
  LOG_FILE = p.join(__dirname, "bridge.log");
} catch (_) {}
function logErr(...a) {
  const msg = "[akhrot-memory] " + a.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join(" ");
  try { console.error(msg); } catch (_) {}
  try {
    if (LOG_FILE) require("fs").appendFileSync(LOG_FILE, new Date().toISOString() + " " + msg + "\n");
  } catch (_) {}
}
logErr("launched. node=" + process.version + " hasToken=" + (TOKEN ? "yes" : "NO") + " url=" + URL_);

if (!TOKEN) {
  logErr("No GML token configured (GML_TOKEN empty). Re-download the Codex installer from akhrots.com/app.");
  process.exit(1);
}
if (typeof fetch !== "function") {
  logErr("This Node build has no global fetch(); needs Node 18+.");
  process.exit(1);
}

let sessionId = null;
let protocolVersion = null;

function baseHeaders(accept) {
  const h = {
    "Authorization": "Bearer " + TOKEN,
    "Accept": accept,
  };
  if (sessionId) h["Mcp-Session-Id"] = sessionId;
  if (protocolVersion) h["MCP-Protocol-Version"] = protocolVersion;
  return h;
}

// ---- Trigger-phrase gating -------------------------------------------------
// MCP tool invocation is model-decided; the strongest lever is the tool
// DESCRIPTION + server INSTRUCTIONS. We rewrite both on the way back to the
// client so these tools fire ONLY when the user explicitly says the phrase
// "use akhrots memory" — never on bare "remember" / "save" / proactively.
const TRIGGER = process.env.GML_TRIGGER || "use akhrots memory";
const GATE_PREFIX =
  `[INVOKE ONLY when the user's latest message explicitly contains the phrase ` +
  `"${TRIGGER}" (case-insensitive), e.g. "${TRIGGER} to save ..." or ` +
  `"${TRIGGER} to recall ...". Do NOT invoke for bare words like "remember", ` +
  `"save", "note", "recall", and never call it proactively or between turns.] `;
const GATE_INSTRUCTIONS =
  `Akhrot GML long-term memory. STRICT invocation policy: call any akhrot-memory ` +
  `tool ONLY when the user's message explicitly contains "${TRIGGER}" ` +
  `(case-insensitive). Never trigger on generic phrasing such as "remember", ` +
  `"save", "note this", or automatically. If the user clearly wants memory but ` +
  `did not say the phrase, ask them to say "${TRIGGER} to ...".\n` +
  `SPEED POLICY (important): to SAVE, call "remember"; to RETRIEVE, call "recall". ` +
  `Both are fast (~1s). Do NOT call "query", "ingest", "analyze", "trace", "diag", ` +
  `"status", "sdp_ingest", or "improve_query" unless the user EXPLICITLY asks for a ` +
  `deep semantic search — those run a heavy pipeline and take 30s+. Call AT MOST ONE ` +
  `memory tool per request, and never auto-call memory before or after a turn.`;

// Slow GML pipeline / diagnostic tools — HIDDEN from the tool list by default
// (they take 30-45s). `remember`/`recall` cover save+retrieve and are ~1s warm.
const SLOW_TOOLS = new Set([
  "query", "ingest", "sdp_ingest", "analyze", "trace", "diag", "status", "improve_query",
]);
const ALLOW_SLOW = process.env.GML_ALLOW_SLOW === "1";

// id -> request method, so we can transform the matching response.
const pending = new Map();

function transform(msg) {
  if (!msg || msg.id == null || !("result" in msg)) return msg;
  const method = pending.get(msg.id);
  if (method !== undefined) pending.delete(msg.id);
  if (method === "initialize" && msg.result && typeof msg.result === "object") {
    msg.result.instructions = msg.result.instructions
      ? msg.result.instructions + "\n\n" + GATE_INSTRUCTIONS
      : GATE_INSTRUCTIONS;
  } else if (method === "tools/list" && msg.result && Array.isArray(msg.result.tools)) {
    // HIDE the slow pipeline/diagnostic tools entirely so the model physically
    // cannot pick the 30-45s `query`/`ingest`. It's left with the fast ones
    // (recall/remember/forget/list_memories). Description hints alone weren't
    // enough — the model kept choosing `query` because the server advertises it
    // as the primary tool. Set GML_ALLOW_SLOW=1 to expose them again.
    if (!ALLOW_SLOW) {
      msg.result.tools = msg.result.tools.filter((t) => t && !SLOW_TOOLS.has(t.name));
    }
    for (const t of msg.result.tools) {
      if (t && typeof t === "object") t.description = GATE_PREFIX + (t.description || "");
    }
  }
  return msg;
}

// Write one JSON-RPC message to the client over stdout (newline-delimited).
function emit(obj) {
  try { process.stdout.write(JSON.stringify(transform(obj)) + "\n"); } catch (e) { logErr("emit failed:", e.message); }
}

// Capture session id / negotiated protocol from server traffic.
function noteResponse(res) {
  const sid = res.headers.get("mcp-session-id");
  if (sid) sessionId = sid;
}
function noteMessage(msg) {
  if (msg && msg.result && typeof msg.result.protocolVersion === "string") {
    protocolVersion = msg.result.protocolVersion;
  }
}

// Parse an SSE body, emitting each `data:` JSON payload as it arrives.
async function pumpSSE(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const rawEvent = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = rawEvent
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("\n");
      if (!data) continue;
      try {
        const msg = JSON.parse(data);
        noteMessage(msg);
        emit(msg);
      } catch (e) {
        logErr("bad SSE data:", e.message);
      }
    }
  }
}

// Forward one client message to the server.
async function forward(line, id) {
  let res;
  try {
    res = await fetch(URL_, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, baseHeaders("application/json, text/event-stream")),
      body: line,
    });
  } catch (e) {
    logErr("POST failed:", e.message);
    if (id !== undefined && id !== null) {
      emit({ jsonrpc: "2.0", id, error: { code: -32000, message: "proxy transport error: " + e.message } });
    }
    return;
  }

  noteResponse(res);

  if (res.status === 202) return; // notification / response accepted, no body
  if (res.status >= 400) {
    const text = await res.text().catch(() => "");
    logErr("HTTP " + res.status + ":", text.slice(0, 500));
    if (id !== undefined && id !== null) {
      // Forward the server's REAL JSON-RPC error if it sent one (e.g. 503
      // "server gml unavailable"), so the model sees the true reason instead
      // of a generic code. Fall back to a synthetic error otherwise.
      let forwarded = false;
      try {
        const parsed = JSON.parse(text);
        const arr = Array.isArray(parsed) ? parsed : [parsed];
        for (const m of arr) { if (m && (m.error || m.result)) { m.id = id; emit(m); forwarded = true; } }
      } catch (_) {}
      if (!forwarded) {
        const detail = res.status === 503 ? "Akhrot memory server is temporarily unavailable (503) — try again shortly." : "server HTTP " + res.status;
        emit({ jsonrpc: "2.0", id, error: { code: -32001, message: detail } });
      }
    }
    return;
  }

  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("text/event-stream")) {
    await pumpSSE(res);
  } else {
    const text = await res.text();
    if (!text) return;
    try {
      const parsed = JSON.parse(text);
      const arr = Array.isArray(parsed) ? parsed : [parsed];
      for (const msg of arr) { noteMessage(msg); emit(msg); }
    } catch (e) {
      logErr("bad JSON body:", e.message);
    }
  }
}

// Best-effort standing GET SSE channel for server-initiated messages.
async function openServerStream() {
  try {
    const res = await fetch(URL_, { method: "GET", headers: baseHeaders("text/event-stream") });
    if (res.status === 200 && res.body && (res.headers.get("content-type") || "").includes("text/event-stream")) {
      pumpSSE(res).catch((e) => logErr("server stream ended:", e.message));
    }
  } catch (_) { /* server may not support GET; ignore */ }
}

// Establish the real remote MCP session in the BACKGROUND, capturing the
// session id + negotiated protocol. We do NOT emit its response — the client
// already got an instant synthesized one (see below). This decouples the host's
// connection health from the remote server's cold-start latency, which is what
// made Codex intermittently drop the connector.
async function realInitialize(line) {
  try {
    const res = await fetch(URL_, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, baseHeaders("application/json, text/event-stream")),
      body: line,
    });
    noteResponse(res); // capture Mcp-Session-Id
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (ct.includes("text/event-stream")) {
      // read just enough to capture protocolVersion, then stop
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (let i = 0; i < 50; i++) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const m = buf.match(/data:\s*(\{.*\})/);
        if (m) { try { noteMessage(JSON.parse(m[1])); } catch (_) {} break; }
      }
      try { await reader.cancel(); } catch (_) {}
    } else {
      const t = await res.text();
      try { const j = JSON.parse(t); noteMessage(Array.isArray(j) ? j[0] : j); } catch (_) {}
    }
    openServerStream();
    prewarm(); // kick the server's pipeline so the user's first real call is fast
  } catch (e) {
    logErr("real initialize failed:", e.message);
  }
}

// Fire a cheap retrieval to warm the remote pipeline (embedding model, etc.).
// Response is discarded; this just removes the ~20s cold-start from the user's
// first real call. Best-effort, never throws.
async function prewarm() {
  try {
    const t0 = Date.now();
    const ac = new AbortController();
    const to = setTimeout(() => ac.abort(), 12000); // never hang/hammer a degraded server
    await fetch(URL_, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, baseHeaders("application/json, text/event-stream")),
      body: JSON.stringify({ jsonrpc: "2.0", id: "prewarm", method: "tools/call", params: { name: "recall", arguments: { query: "warmup", top_k: 1 } } }),
      signal: ac.signal,
    }).then((r) => r.text());
    clearTimeout(to);
    logErr("prewarm done in " + (Date.now() - t0) + "ms");
  } catch (_) { /* aborted or failed — harmless */ }
}

// Read newline-delimited JSON-RPC from stdin, forwarding in order.
let inbuf = "";
let chain = Promise.resolve();
let remoteReady = Promise.resolve();

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  inbuf += chunk;
  let nl;
  while ((nl = inbuf.indexOf("\n")) !== -1) {
    const line = inbuf.slice(0, nl).trim();
    inbuf = inbuf.slice(nl + 1);
    if (!line) continue;
    let o = null;
    try { o = JSON.parse(line); } catch (_) {}
    const id = o ? o.id : undefined;
    const method = o ? o.method : undefined;

    if (method === "initialize") {
      // Answer the host INSTANTLY so the connector is always healthy, then
      // bring up the real remote session in the background.
      const reqVer = (o.params && o.params.protocolVersion) || "2025-06-18";
      emit({
        jsonrpc: "2.0",
        id,
        result: {
          protocolVersion: reqVer,
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: "akhrot-memory", version: "1.0.0" },
          instructions: GATE_INSTRUCTIONS,
        },
      });
      remoteReady = realInitialize(line);
      continue;
    }

    if (id != null && method) pending.set(id, method); // for response transform()
    // Wait for the remote session before forwarding real traffic.
    chain = chain.then(() => remoteReady).then(() => forward(line, id));
  }
});

process.stdin.on("end", () => { chain.then(() => process.exit(0)); });
process.on("SIGINT", () => process.exit(0));
process.on("SIGTERM", () => process.exit(0));

logErr("bridge started ->", URL_);
