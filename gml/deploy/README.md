# Deploying GML to a VM

End-to-end runbook for putting the orchestration stack live on a Debian/Ubuntu VM. Adapt paths and hostnames to your reality.

## Quick start (one command, native — no Docker)

`deploy/vm-bootstrap.sh` does the whole thing: installs Postgres + pgvector, creates the `gml` DB + `gml_app` role and applies the schema, builds a Python venv, installs GML, and runs the API as a `systemd` service (`gml-api`). SAM/LLM (Ollama) is opt-in.

```bash
git clone https://github.com/tronocitylabs/gmlcore.git && cd gmlcore
git checkout feat/gml-web-ui-api

sudo bash deploy/vm-bootstrap.sh --no-sam --issue-user me   # API + Postgres (heuristic)
# or, with the local SAM/extraction LLM (installs Ollama, pulls qwen2.5:3b):
sudo bash deploy/vm-bootstrap.sh --sam --issue-user me
# add --domain gml.example.com --email you@x.com to also wire nginx + TLS
```

Secrets land in `/etc/gml/` (DB password, master API key, `gml.env`). Manage the service with `systemctl status|restart gml-api` and `journalctl -u gml-api -f`. The manual steps below document what the script automates.

## What gets deployed

```
                ┌─────────────────────────────────────────────────┐
                │  nginx :443 (TLS termination, rate limits)      │
                └──┬────────────────────┬────────────────────┬────┘
                   │                    │                    │
        /api/*  ▼                   /mcp  ▼            /         ▼
       ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
       │ FastAPI :8000    │   │ FastMCP :8765    │   │ Next.js UI build │
       │ orchestration    │   │ streamable-http  │   │ (static export)  │
       │ .server          │   │ orchestration    │   │ /var/www/gml-ui  │
       │                  │   │ .mcp_server      │   │                  │
       └────────┬─────────┘   └────────┬─────────┘   └──────────────────┘
                │                      │
                └──────┬───────────────┘
                       ▼
              ~/.gml/users.jsonl   (shared per-user key store)
              ~/.gml/memories.jsonl (shared memory store)
              /opt/gml/models      (FT'd Qwen LoRA, FT'd CE)
```

Both backends authenticate against the same `users.jsonl`, so a key issued from FastAPI's admin endpoint works in MCP and vice versa.

## Prereqs on the GCP VM

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
    nginx git build-essential certbot python3-certbot-nginx

# Create the service user
sudo useradd -r -m -s /bin/bash gml
sudo mkdir -p /opt/gml /opt/gml/data /var/www/gml-ui
sudo chown -R gml:gml /opt/gml
```

For GPU support (much faster than CPU): install CUDA + the matching PyTorch build. See https://pytorch.org/get-started for the exact wheel URL for your CUDA version.

## Pull the code + install

```bash
sudo -u gml bash <<'EOF'
cd /opt/gml
git clone <your-repo-url> .
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# install bench/FT extras if you intend to run benches on the box
pip install fastembed sentence-transformers peft transformers torch
EOF
```

## Bring the FT'd models over

The model directories (`models/qwen_locomo_ft`, `models/ce_locomo_ft`) are excluded from git. Copy them up:

```bash
# from your laptop
rsync -avz models/ gml@your-box-ip:/opt/gml/models/
```

## Install the services

```bash
sudo cp /opt/gml/deploy/systemd-gml-api.service /etc/systemd/system/gml-api.service
sudo cp /opt/gml/deploy/systemd-gml-mcp.service /etc/systemd/system/gml-mcp.service

# Edit each to set GML_API_KEY=<your master key> and adjust paths.
sudo systemctl edit --full gml-api.service   # opens the file for editing
sudo systemctl edit --full gml-mcp.service

sudo systemctl daemon-reload
sudo systemctl enable --now gml-api gml-mcp
sudo systemctl status gml-api gml-mcp
```

## Wire up nginx

`deploy/nginx.conf` is a generic single-host template (`gml.example.com`). The
**live production config** for the akhrots.com deployment is captured verbatim in
`deploy/nginx/` and is the source of truth for the running VM:

- `deploy/nginx/akhrots.com.conf` — single public entry point. `/` + `/app` +
  `/login` → Next.js (`:3000`), `/api/*` + `/auth/*` → FastAPI (`:8000`),
  `/mcp` + `/relay/*` → MCP relay (`:8080`). The Next.js dashboard (`/app`,
  `/app/trace`) is the only visualization surface; FastAPI serves no HTML viz.
- `deploy/nginx/mcp.akhrots.com.conf` — legacy subdomain, 308-redirects to
  `akhrots.com`. **MCP clients must point directly at `https://akhrots.com/mcp`**
  (cross-host redirects strip the `Authorization` header).

```bash
# Generic template:
sudo cp /opt/gml/deploy/nginx.conf /etc/nginx/sites-available/gml
sudo sed -i 's/gml\.example\.com/your.real.host/g' /etc/nginx/sites-available/gml
sudo ln -s /etc/nginx/sites-available/gml /etc/nginx/sites-enabled/

# Or the live akhrots.com configs:
sudo cp /opt/gml/deploy/nginx/akhrots.com.conf     /etc/nginx/sites-available/akhrot
sudo cp /opt/gml/deploy/nginx/mcp.akhrots.com.conf /etc/nginx/sites-available/mcp
sudo ln -sf /etc/nginx/sites-available/akhrot /etc/nginx/sites-enabled/
sudo ln -sf /etc/nginx/sites-available/mcp    /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl reload nginx
```

## Get a TLS cert

```bash
sudo certbot --nginx -d your.real.host
# Certbot will edit the nginx config in place to point to the right cert files.
```

## Issue the first user key

```bash
# Pick a strong master key
MASTER="$(openssl rand -hex 32)"

# Update both systemd units with this value, then restart them.
sudo systemctl edit --full gml-api.service     # set GML_API_KEY=$MASTER
sudo systemctl edit --full gml-mcp.service     # same value
sudo systemctl restart gml-api gml-mcp

# Issue a key for yourself
curl -X POST https://your.real.host/api/admin/keys \
  -H "X-API-Key: $MASTER" \
  -H "content-type: application/json" \
  -d '{"user_id":"atharva","label":"laptop"}'
# Save the returned `key`.
```

## Smoke test

```bash
KEY="gml_xxx..."   # your user key

# REST surface
curl https://your.real.host/api/health
curl https://your.real.host/api/memories -H "X-API-Key: $KEY"

# Memory ingest (regex path — fastest)
curl -X POST https://your.real.host/api/memory/sdp_ingest \
  -H "X-API-Key: $KEY" \
  -H "content-type: application/json" \
  -d '{"user_query":"Hi","assistant_reply":"We use Adyen for payments."}'

# Recall
curl -X POST https://your.real.host/api/memory/recall \
  -H "X-API-Key: $KEY" \
  -H "content-type: application/json" \
  -d '{"query":"what do we use for payments","top_k":5}'

# MCP surface (more complex JSON-RPC; the platform clients handle this)
# A successful handshake looks like a `tools/list` response with 12 tools.
```

## Adding new users

Same admin call — repeat per user, send them the returned `key` over a secure channel (Signal, password manager, whatever your company uses). They paste it into the platform-specific config from `docs/MCP_SETUP.md`.

## Logs

```bash
sudo journalctl -u gml-api -f
sudo journalctl -u gml-mcp -f
sudo tail -f /var/log/nginx/{access,error}.log
```

## Updating

```bash
sudo -u gml bash <<'EOF'
cd /opt/gml
git pull --ff-only
.venv/bin/pip install -e .   # in case deps changed
EOF
sudo systemctl restart gml-api gml-mcp
```

## What's intentionally NOT in this deploy

- **Multi-process workers** for FastAPI. `--workers 1` is the safe default because the in-memory `HybridRetriever` is process-local — workers would have inconsistent views of the memory store. For real horizontal scale, swap `JsonlMemoryStore` for SQLite + sqlite-vec and use Uvicorn workers behind a shared file.
- **DB**: `users.jsonl` is a JSONL file with a process-local lock. Fine for low-write workloads (key issuance is rare). Migrate to SQLite when you need concurrent writes.
- **Per-user memory isolation**: today, all authenticated users share the same memory pool. The key infrastructure is in place; the next sprint will namespace `~/.gml/memories-{user_id}.jsonl` per user.
- **OAuth**: per-user API keys are the auth primitive. OAuth flow (Google/GitHub sign-in → key issuance) is a follow-up sprint.
