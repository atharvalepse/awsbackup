#!/usr/bin/env bash
# =============================================================================
# vm-bootstrap.sh — native (no Docker) deploy of GML on a Debian/Ubuntu VM.
#
# Installs Postgres + pgvector, creates the gml DB + gml_app role + applies the
# schema, builds a Python venv, installs GML, and runs the API as a systemd
# service. SAM/LLM (Ollama) is optional.
#
#   sudo bash deploy/vm-bootstrap.sh --no-sam                 # API + Postgres only
#   sudo bash deploy/vm-bootstrap.sh --sam                    # + Ollama for SAM
#   sudo bash deploy/vm-bootstrap.sh --no-sam --issue-user me
#   sudo bash deploy/vm-bootstrap.sh --sam --domain gml.example.com --email you@x.com
#
# Idempotent-ish: re-running reuses the generated secrets and reconciles the
# SAM mode + restarts the service. Run as root (via sudo).
# =============================================================================
set -euo pipefail

# ---- args -------------------------------------------------------------------
SAM=0
PORT=8000
DB_NAME=gml
SKIP_SYS_DEPS=0
OLLAMA_MODEL="qwen2.5:3b"
ISSUE_USER=""
DOMAIN=""
EMAIL=""

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --sam)             SAM=1 ;;
        --no-sam)          SAM=0 ;;
        --port)            PORT="${2:?}"; shift ;;
        --db-name)         DB_NAME="${2:?}"; shift ;;
        --ollama-model)    OLLAMA_MODEL="${2:?}"; shift ;;
        --issue-user)      ISSUE_USER="${2:?}"; shift ;;
        --domain)          DOMAIN="${2:?}"; shift ;;
        --email)           EMAIL="${2:?}"; shift ;;
        --skip-system-deps) SKIP_SYS_DEPS=1 ;;
        -h|--help)         usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
    shift
done

# ---- preflight --------------------------------------------------------------
[ "$(id -u)" -eq 0 ] || { echo "Run as root: sudo bash $0 ..." >&2; exit 1; }
command -v apt-get >/dev/null || { echo "This script targets Debian/Ubuntu (apt)." >&2; exit 1; }
command -v systemctl >/dev/null || { echo "systemd is required." >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVC_USER="${SUDO_USER:-root}"
[ "$SVC_USER" = "root" ] && echo "⚠ no SUDO_USER — the service will run as root (prefer: sudo bash ... as a normal user)."
GML_ETC=/etc/gml
ENV_FILE="$GML_ETC/gml.env"
DBPW_FILE="$GML_ETC/db-password"
APIKEY_FILE="$GML_ETC/api-key"

log()  { printf '\033[1;36m▶ %s\033[0m\n' "$*"; }
asusr() { sudo -u "$SVC_USER" -H "$@"; }   # run as the service user with its HOME

mkdir -p "$GML_ETC"; chmod 750 "$GML_ETC"

# ---- 1. system packages -----------------------------------------------------
if [ "$SKIP_SYS_DEPS" = "0" ]; then
    log "Installing system packages (Postgres, build tools, Python)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential openssl \
        postgresql postgresql-contrib \
        python3 python3-venv python3-dev
fi

# Ensure a Python >= 3.11 (the project requires it). Ubuntu 22.04 ships 3.10 —
# pull 3.12 from deadsnakes there.
pyok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null; }
PYBIN=""
for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null && pyok "$c"; then PYBIN="$(command -v "$c")"; break; fi
done
if [ -z "$PYBIN" ]; then
    log "System Python < 3.11 — installing python3.12 from deadsnakes"
    apt-get install -y --no-install-recommends software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
    PYBIN="$(command -v python3.12)"
fi
log "Using Python: $PYBIN ($("$PYBIN" --version))"

# ---- 2. pgvector ------------------------------------------------------------
systemctl enable --now postgresql
PGVER="$(ls /usr/lib/postgresql/ 2>/dev/null | sort -n | tail -1)"
[ -n "$PGVER" ] || { echo "Could not detect installed Postgres version." >&2; exit 1; }
if sudo -u postgres psql -d postgres -tAc \
     "SELECT 1 FROM pg_available_extensions WHERE name='vector'" 2>/dev/null | grep -q 1; then
    log "pgvector already available for Postgres $PGVER"
else
    log "Installing pgvector for Postgres $PGVER"
    if ! apt-get install -y "postgresql-${PGVER}-pgvector" 2>/dev/null; then
        log "  apt package unavailable — building pgvector from source"
        apt-get install -y --no-install-recommends "postgresql-server-dev-${PGVER}" git build-essential
        tmp="$(mktemp -d)"
        git clone --depth 1 --branch v0.8.0 https://github.com/pgvector/pgvector.git "$tmp/pgvector"
        make -C "$tmp/pgvector" -j"$(nproc)"
        make -C "$tmp/pgvector" install
        rm -rf "$tmp"
    fi
fi

# ---- 3. database (RLS-correct: tables owned by superuser, app role granted) -
[ -f "$DBPW_FILE" ] || { openssl rand -hex 24 > "$DBPW_FILE"; chmod 600 "$DBPW_FILE"; }
DB_PASSWORD="$(cat "$DBPW_FILE")"

log "Creating role gml_app + database '$DB_NAME'"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gml_app') THEN
        CREATE ROLE gml_app LOGIN PASSWORD '${DB_PASSWORD}';
    ELSE
        ALTER ROLE gml_app PASSWORD '${DB_PASSWORD}';
    END IF;
END
\$\$;
SELECT 'CREATE DATABASE ${DB_NAME}'
 WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec
SQL

# Apply the consolidated schema AS the postgres superuser so the tables are
# owned by postgres (not gml_app). gml_app is then a non-owner the RLS policies
# actually constrain — table owners bypass RLS unless FORCE'd.
log "Applying schema (migrations/all_in_one.sql) as superuser"
sudo -u postgres psql -d "$DB_NAME" -v ON_ERROR_STOP=1 -f "$REPO_ROOT/migrations/all_in_one.sql"

DB_URL="postgresql://gml_app:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}"

# ---- 4. Python venv + GML ---------------------------------------------------
log "Building venv + installing GML at $REPO_ROOT/.venv"
asusr "$PYBIN" -m venv "$REPO_ROOT/.venv"
asusr "$REPO_ROOT/.venv/bin/pip" install -q -U pip
asusr "$REPO_ROOT/.venv/bin/pip" install -q -e "$REPO_ROOT"
# Pre-warm the FastEmbed model so the first request isn't a cold download.
asusr "$REPO_ROOT/.venv/bin/python" -c \
    "from orchestration.embedder import FastEmbedEmbedder; FastEmbedEmbedder()" || \
    log "  (FastEmbed pre-warm failed — it will download on first request)"

# ---- 5. Ollama (only with --sam) -------------------------------------------
if [ "$SAM" = "1" ]; then
    if ! command -v ollama >/dev/null; then
        log "Installing Ollama"
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    systemctl enable --now ollama || true
    log "Pulling Ollama model: $OLLAMA_MODEL (this can take a while on CPU)"
    ollama pull "$OLLAMA_MODEL" || log "  (model pull failed — SAM will degrade until it's available)"
fi

# ---- 6. env file + systemd service -----------------------------------------
[ -f "$APIKEY_FILE" ] || { openssl rand -hex 32 > "$APIKEY_FILE"; chmod 600 "$APIKEY_FILE"; }
API_KEY="$(cat "$APIKEY_FILE")"

log "Writing $ENV_FILE"
cat > "$ENV_FILE" <<EOF
GML_STORAGE_BACKEND=postgres
GML_DATABASE_URL=${DB_URL}
GML_API_KEY=${API_KEY}
GML_OLLAMA_BASE_URL=http://127.0.0.1:11434
GML_OLLAMA_MODEL=${OLLAMA_MODEL}
GML_CORS_ORIGINS=*
EOF
chmod 600 "$ENV_FILE"

SAM_FLAG=""; SAM_AFTER=""
if [ "$SAM" = "0" ]; then
    SAM_FLAG=" --no-sam-llm"
else
    SAM_AFTER=" ollama.service"
fi

log "Writing + starting systemd service gml-api"
cat > /etc/systemd/system/gml-api.service <<EOF
[Unit]
Description=GML orchestration API
After=network-online.target postgresql.service${SAM_AFTER}
Wants=network-online.target postgresql.service

[Service]
Type=simple
User=${SVC_USER}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=${ENV_FILE}
ExecStart=${REPO_ROOT}/.venv/bin/gml serve --host 127.0.0.1 --port ${PORT}${SAM_FLAG}
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable gml-api
systemctl restart gml-api

# ---- 7. wait for health -----------------------------------------------------
log "Waiting for the API on 127.0.0.1:${PORT} ..."
deadline=$(( $(date +%s) + 120 ))
until curl -fs "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; do
    if ! systemctl is-active --quiet gml-api; then
        echo "gml-api failed to start. Recent logs:" >&2
        journalctl -u gml-api -n 40 --no-pager >&2 || true
        exit 1
    fi
    [ "$(date +%s)" -ge "$deadline" ] && { echo "API not healthy in time:" >&2; journalctl -u gml-api -n 40 --no-pager >&2; exit 1; }
    sleep 3
done
log "API healthy: $(curl -s "http://127.0.0.1:${PORT}/api/health")"

# ---- 8. optional: issue a first user key ------------------------------------
if [ -n "$ISSUE_USER" ]; then
    RESP="$(curl -s -X POST "http://127.0.0.1:${PORT}/api/admin/keys" \
        -H "X-API-Key: ${API_KEY}" -H 'content-type: application/json' \
        -d "{\"user_id\":\"${ISSUE_USER}\",\"label\":\"bootstrap\"}")"
    USER_KEY="$(printf '%s' "$RESP" | sed -n 's/.*"key":"\([^"]*\)".*/\1/p')"
    [ -n "$USER_KEY" ] && log "Issued API key for '${ISSUE_USER}': ${USER_KEY}" \
                       || echo "Key issuance response: $RESP" >&2
fi

# ---- 9. optional: nginx + TLS ----------------------------------------------
if [ -n "$DOMAIN" ]; then
    log "Configuring nginx + TLS for ${DOMAIN}"
    apt-get install -y --no-install-recommends nginx certbot python3-certbot-nginx
    cp "$REPO_ROOT/deploy/nginx.conf" /etc/nginx/sites-available/gml
    sed -i "s/gml\.example\.com/${DOMAIN}/g" /etc/nginx/sites-available/gml
    ln -sf /etc/nginx/sites-available/gml /etc/nginx/sites-enabled/gml
    nginx -t && systemctl reload nginx
    if [ -n "$EMAIL" ]; then
        certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
        log "TLS provisioned — https://${DOMAIN}"
    else
        log "nginx configured. Finish TLS with: sudo certbot --nginx -d ${DOMAIN}"
    fi
fi

# ---- 10. summary ------------------------------------------------------------
cat <<EOF

──────────────────────────────────────────────────────────────────────
 GML is up (native, no Docker).  SAM/LLM: $([ "$SAM" = 1 ] && echo enabled || echo "disabled (heuristic)")
   Service    : systemctl status gml-api    |    logs: journalctl -u gml-api -f
   Health     : http://127.0.0.1:${PORT}/api/health
   DB         : ${DB_NAME} (role gml_app; password in ${DBPW_FILE})
   Master key : ${APIKEY_FILE}   (env in ${ENV_FILE})
   Restart    : sudo systemctl restart gml-api
$([ -z "$DOMAIN" ] && echo "   Public TLS : not configured — re-run with --domain (and --email), or see deploy/README.md")
──────────────────────────────────────────────────────────────────────
EOF
