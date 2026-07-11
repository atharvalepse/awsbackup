#!/usr/bin/env bash
# Install Postgres 16 + pgvector + pg_trgm on the GML VM and create the
# `gml` database with the `gml_app` role. Run AS ROOT on the VM:
#
#   sudo bash 02-install-postgres.sh
#
# Idempotent — safe to re-run. The script:
#   1. apt-installs Postgres 16 (+ contrib for trigram search)
#   2. compiles + installs pgvector from source (Debian repos lag)
#   3. creates the `gml` DB and the `gml_app` role
#   4. writes a strong password to /root/.gml-db-password (chmod 600)
#   5. enables required extensions + applies schema.sql if present

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo bash $0" >&2
    exit 1
fi

# ----------------------------------------------------------------------
echo "▶ apt update + base packages"
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release \
    postgresql-16 postgresql-contrib-16 \
    postgresql-server-dev-16 \
    build-essential git

# ----------------------------------------------------------------------
echo "▶ Installing pgvector from source (Debian 12 repos lag; this gives us latest)"
PGVECTOR_VERSION="${PGVECTOR_VERSION:-v0.8.0}"
if ! find /usr/lib/postgresql -name vector.so 2>/dev/null | grep -q vector.so; then
    cd /tmp
    rm -rf pgvector
    git clone --branch "$PGVECTOR_VERSION" https://github.com/pgvector/pgvector.git
    cd pgvector
    make -j"$(nproc)"
    make install
    echo "  pgvector $PGVECTOR_VERSION installed"
else
    echo "  pgvector already installed — skipping build"
fi

# ----------------------------------------------------------------------
echo "▶ Postgres tuning for a 4 GB VM"
PG_CONF=/etc/postgresql/16/main/postgresql.conf
# Idempotent updates — set or replace these lines.
update_pg_conf() {
    local key="$1" val="$2"
    if grep -qE "^\s*#?\s*${key}\s*=" "$PG_CONF"; then
        sed -i "s|^\s*#\?\s*${key}\s*=.*|${key} = ${val}|" "$PG_CONF"
    else
        echo "${key} = ${val}" >> "$PG_CONF"
    fi
}
update_pg_conf shared_buffers "1GB"          # ~25% of RAM
update_pg_conf work_mem "32MB"
update_pg_conf maintenance_work_mem "256MB"
update_pg_conf effective_cache_size "3GB"    # ~75% of RAM
update_pg_conf max_connections "50"          # plenty for <20 users
update_pg_conf listen_addresses "'127.0.0.1'"   # NOT public — nginx → API → DB

systemctl restart postgresql

# ----------------------------------------------------------------------
echo "▶ Creating gml DB + role"
DB_PASSWORD_FILE=/root/.gml-db-password
if [ ! -f "$DB_PASSWORD_FILE" ]; then
    openssl rand -hex 24 > "$DB_PASSWORD_FILE"
    chmod 600 "$DB_PASSWORD_FILE"
fi
DB_PASSWORD=$(cat "$DB_PASSWORD_FILE")

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

SELECT 'CREATE DATABASE gml OWNER gml_app'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gml')\gexec
SQL

# Extensions go inside the db, not at cluster level
sudo -u postgres psql -d gml -v ON_ERROR_STOP=1 <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()
GRANT ALL PRIVILEGES ON SCHEMA public TO gml_app;
SQL

# ----------------------------------------------------------------------
MIGRATIONS_DIR="$(dirname "$0")/../../migrations"
if [ -d "$MIGRATIONS_DIR" ]; then
    echo "▶ Applying migrations from $MIGRATIONS_DIR"
    GML_DATABASE_URL="postgresql://gml_app:${DB_PASSWORD}@127.0.0.1/gml" \
        bash "$MIGRATIONS_DIR/apply_all.sh"
else
    echo "  (migrations/ dir not found — apply manually with apply_all.sh)"
fi

# ----------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Postgres ready."
echo ""
echo "  DB:        gml"
echo "  Role:      gml_app"
echo "  Password:  saved in $DB_PASSWORD_FILE (chmod 600)"
echo ""
echo "  Connect from the same VM:"
echo "    psql 'postgresql://gml_app:'\"\$(cat $DB_PASSWORD_FILE)\"'@127.0.0.1/gml'"
echo ""
echo "  Add this to your systemd unit environment:"
echo "    GML_DATABASE_URL=postgresql://gml_app:<password>@127.0.0.1/gml"
echo "═══════════════════════════════════════════════════════════════"
