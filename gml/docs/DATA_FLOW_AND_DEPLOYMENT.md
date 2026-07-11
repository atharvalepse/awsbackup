# GML — Data Flow, Storage & MCP (How everything connects)

This doc explains how the deployed system fits together: the services, the
databases, how a memory is written and read, how multi-tenancy works, and how
an external tool (Claude/Cursor) reaches the server through MCP.

For *what each code component does* (SDP, SAM, AAL, reranker, …) see
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## 1. Deployment topology (the live VM)

GCP VM `e2-standard-4` (4 vCPU / 16 GB). Two disks: a 10 GB root (`/`) and a
40 GB data disk mounted at `/opt/gml` (everything heavy lives here — venv,
models, Ollama, Postgres data is on root).

```
                          Internet
                             │  (HTTPS, Let's Encrypt)
                    ┌────────▼─────────┐
                    │  nginx :80/:443  │   mcp.akhrots.com  → relay
                    │   (TLS, proxy)   │   akhrots.com      → static site
                    └───┬─────────┬────┘
        /mcp (relay)    │         │   /api, / (gml site, :80)
                 ┌──────▼───┐  ┌──▼──────────────┐
                 │ mcp-relay│  │ gml-api (FastAPI│   :8000
                 │  :8080   │  │  uvicorn)       │   REST + /auth + admin
                 └────┬─────┘  └──┬──────────────┘
   downlink/uplink    │           │  (shares the pipeline + stores)
                 ┌────▼─────────┐ │
                 │ gml-connector│ │   spawns one `gml mcp` (stdio) per session
                 └────┬─────────┘ │
                 ┌────▼───────────▼────┐        ┌─────────────────┐
                 │  GML pipeline        │◀──────▶│ Ollama :11434   │
                 │  (embed→retrieve→    │  SAM   │ gml-qwen-ft     │
                 │   rerank→SAM→assemble│  LLM   │ (FT Qwen Q5_K_M)│
                 └────┬───────────┬─────┘        └─────────────────┘
            embeddings│           │ SQL
        ┌─────────────▼──┐   ┌────▼──────────────────────────────┐
        │ FT embedder +  │   │ PostgreSQL :5432                  │
        │ FT cross-enc   │   │  • db "gml": memories(pgvector),  │
        │ (torch/ST,CPU) │   │    users, user_keys  (+ RLS)      │
        └────────────────┘   │  • db "mcp_relay": relay users +  │
                             │    api_tokens                     │
                             └───────────────────────────────────┘
```

### systemd services
| Service | Port | Role |
|---|---|---|
| `postgresql` | 5432 | `gml` DB (memories/users/keys) + `mcp_relay` DB (relay auth) |
| `ollama` | 11434 | serves `gml-qwen-ft` (FT Qwen, kept warm via `OLLAMA_KEEP_ALIVE=-1`) |
| `gml-api` | 8000 | FastAPI: REST `/api/*`, `/auth/*`, admin keys, `/viz` |
| `mcp-relay` | 8080 | broker: hosts dial `/mcp`, connectors dial `/relay/*`; Postgres token auth + `/dashboard` |
| `gml-connector` | — | registers `gml` with the relay (globally), spawns `gml mcp` per session |
| `nginx` | 80/443 | TLS termination + reverse proxy (`mcp.akhrots.com` → relay) |
| `gml-warm` | — | oneshot on boot: pre-loads the LLM so the first query isn't a cold start |

### Config (env files, root-owned `600`)
- `/etc/gml/gml.env` — used by `gml-api` **and** the connector-spawned `gml mcp`:
  `GML_STORAGE_BACKEND=postgres`, `GML_DATABASE_URL`, `GML_API_KEY` (master),
  `GML_JWT_SECRET`, `GML_EMBEDDER=st`, `GML_ST_EMBED_MODEL=/opt/gml/models/embedder_locomo_ft`,
  `GML_CE_ENSEMBLE=…/ce_locomo_ft|jinaai/jina-reranker-v2-base-multilingual|0.7`,
  `GML_OLLAMA_BASE_URL`, `GML_OLLAMA_MODEL=gml-qwen-ft`, `GML_MCP_USER` (default tenant),
  `HF_HOME=/opt/gml/.hf`.
- `/etc/gml/relay.env` — `DATABASE_URL` (mcp_relay DB), `RELAY_SESSION_SECRET`.
- `/etc/gml/{db-password,api-key,relay-token.txt}` — generated secrets.

### Fine-tuned models (`/opt/gml/models`)
| Dir | What | Served via |
|---|---|---|
| `embedder_locomo_ft` | FT bge-small (384-dim) embedder | `GML_EMBEDDER=st` (sentence-transformers, CPU) |
| `ce_locomo_ft` | FT cross-encoder reranker | `GML_CE_ENSEMBLE` (ensembled with jina-reranker-v2) |
| `qwen_locomo_ft/*.gguf` | FT Qwen 2.5-3B (Q5_K_M used) | imported into Ollama as `gml-qwen-ft` |

---

## 2. The two databases

### `gml` — the memory store (per-tenant)
- **`users`** — `user_id` (PK), `email` (unique), `password_hash`, `plan`,
  `quota_bytes`, `bytes_used`, `is_active`. Every memory must belong to a user.
- **`user_keys`** — API keys → `user_id` (hashed lookup, revocable).
- **`memories`** — the heart:
  - `id`, `user_id` (FK → users, `NOT NULL`), `content`, `entity/attribute/value`,
    `source`, `authority_score`, `pinned`, `timestamp`, `raw_metadata` (JSONB)
  - `embedding vector(384)` — pgvector, **HNSW** cosine index (`<=>`)
  - `content_tsv` — tsvector FTS column (BM25-style `ts_rank_cd`)
  - `aal_simplemem` (TEXT) + `aal_sjson` (JSONB, GIN-indexed) — the canonical AAL views
  - `byte_size` — maintained by a trigger that updates `users.bytes_used` (quota)
- **Row-Level Security**: `memories`/`user_keys` have an RLS policy keyed on
  `app.current_user_id` (set per transaction). A query only sees its tenant's
  rows. `app.is_admin='true'` bypasses (migrations, admin tools, key lookups).
  The app role `gml_app` obeys RLS; tables are owned by the `postgres`
  superuser so the owner-bypass doesn't defeat it.

### `mcp_relay` — the relay's own auth
- `users` (email + password) and `api_tokens` (hashed, revocable). Completely
  separate from the `gml` DB. A relay token → a relay `user_id`, which is the
  routing/isolation key inside the relay.

---

## 3. Write path (saving a memory)

Two ingest flavours, both end in the same place:

```
conversation turn (user_query + assistant_reply)
   │
   ├── LLM path  (/api/memory/ingest, MCP `ingest`)
   │     MemoryExtractor (Ollama LLM) → durable facts
   │
   └── regex path (/api/memory/sdp_ingest, MCP `sdp_ingest`)  ← fast, no LLM
         SDPPipeline → atomic facts
   │
   ▼
AALConverter → AAL records ({simplemem, sjson})  →  MemoryItem[]
   │
   ▼
embedder.embed_batch(content)   →  vector(384)     (FT ST embedder)
   │
   ▼
PostgresMemoryStore.add_many(items, user_id)
   • set app.current_user_id (RLS)
   • auto-provision the users row (ON CONFLICT DO NOTHING)  ← so new tenants don't FK-fail
   • INSERT … memories(content, embedding, aal_simplemem, aal_sjson, byte_size, …)
   • byte-tracking trigger bumps users.bytes_used (quota)
```

The single-fact `remember(content)` MCP tool skips extraction and writes one
MemoryItem directly. Embeddings are computed **on write** — a row with a NULL
embedding would be invisible to vector search.

---

## 4. Read path (recall / query)

```
query text + target model + user_id
   │
   ▼ Classifier        intent/entities (keyword fast-path or LLM)
   ▼ Embedder          FT ST embedder → query vector(384)
   ▼ HybridRetriever   dense (pgvector cosine, HNSW) + sparse (tsvector BM25)
   │                   fused by Reciprocal Rank Fusion (k≈60), RLS-scoped to user
   ├─ nothing found → SAM.reason_from_scratch()  (LLM improves the bare query)
   └─ found → top-50 → Reranker (FT cross-encoder ensemble) → top-10
                     → SAM.resolve_conflicts()  (drop superseded, rewrite query)
                     → Assembler (fit into token budget; full→summary→drop)
                     → Translator (format for the target model family)
                     → TranslatedPayload  (the context block the model receives)
```

- **`/api/memory/recall`** and the MCP `recall` tool take the **fast** branch:
  embed → retrieve → return raw scored hits (no rerank/SAM). Used for quick lookup.
- **`/api/memory/synthesize`** / MCP `query` run the **full** pipeline above
  (rerank + SAM + assemble) — higher quality, but `query` invokes the LLM so
  it's the slow one on CPU.

---

## 5. MCP — how an external tool reaches the server

The relay is a **broker**: clients dial *in*, servers dial *in* via a
connector; neither needs an inbound port except the relay.

```
Claude Desktop / Cursor / any MCP tool
   │  (mcp-remote or mcp-relay-client + RELAY_TOKEN)
   ▼  HTTPS
nginx (mcp.akhrots.com)  →  mcp-relay :8080
   │   • token → relay user_id  (the trust/isolation boundary)
   │   • routes /mcp?server=gml  to the registered "gml" connector
   ▼   downlink envelope {type:"open", session, user:<relay user_id>}
gml-connector
   │   • spawns ONE `gml mcp` (stdio) per host session
   │   • sets GML_MCP_USER=<relay user_id> on that subprocess  ← per-user scoping
   ▼
`gml mcp` (stdio MCP server)  → the GML pipeline (§3/§4), scoped to that tenant
   ▲   replies flow back up: subprocess stdout → connector uplink → relay → host
```

**Per-user isolation:** the connector registered `gml` as a **global** server
(`RELAY_REGISTER_GLOBAL=1`), so any authenticated relay user can reach it; the
relay forwards the host's `user_id` per session; the connector scopes the
backend to it via `GML_MCP_USER`. Result: **token A and token B get separate
memory pools; the same token across Claude + Cursor shares one pool.** (Code on
branch `feat/per-user-gml` of `tronocitylabs/mcp-relay`.)

If no per-session user is supplied (e.g. stdio without the relay), the MCP
falls back to `GML_MCP_USER` from the env (a shared tenant) so writes never
fail for lack of a tenant.

---

## 6. Authentication summary

| Surface | Credential | Resolves to |
|---|---|---|
| REST API (`:8000`) | `X-API-Key` master key, per-user API key, **or** JWT (`/auth/login`) | a `gml` `user_id` (RLS scope) |
| MCP (via relay) | relay bearer token | a relay `user_id` → `GML_MCP_USER` tenant |
| Relay admin/dashboard | email+password → session → API token | relay `user_id` |

Every authenticated request resolves a `user_id`; Postgres RLS then guarantees
the request only ever touches that tenant's memories.

---

## 7. Quick operational reference

```bash
# health (use the domain — the VM's public IP has changed across deploys)
curl https://akhrots.com/api/health
curl https://mcp.akhrots.com/dashboard -o /dev/null -w '%{http_code}\n'

# logs
journalctl -u gml-api -f ; journalctl -u mcp-relay -f ; journalctl -u gml-connector -f

# DB
sudo -u postgres psql -d gml          # memories/users
sudo -u postgres psql -d mcp_relay    # relay auth

# restart
sudo systemctl restart gml-api mcp-relay gml-connector ollama
```
