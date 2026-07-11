#!/usr/bin/env bash
# Apply all pending migrations from this directory in lexical order.
#
# Idempotent: tracks which files have been applied in the schema_migrations
# table (created by 001_extensions.sql). Re-running this is a no-op once
# everything's already applied.
#
# Usage:
#   bash migrations/apply_all.sh "postgresql://gml_app:PASS@127.0.0.1/gml"
# Or set the env var:
#   GML_DATABASE_URL="..." bash migrations/apply_all.sh

set -euo pipefail

DB_URL="${1:-${GML_DATABASE_URL:-}}"
if [ -z "$DB_URL" ]; then
    echo "Usage: $0 <database-url>     or set GML_DATABASE_URL" >&2
    exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"

# Make sure 001 has been applied first (it creates schema_migrations).
psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$DIR/001_extensions.sql" > /dev/null

# Then iterate every numbered SQL file. Skip ones already applied.
shopt -s nullglob
for file in "$DIR"/[0-9][0-9][0-9]_*.sql; do
    base="$(basename "$file")"
    checksum="$(sha256sum "$file" | awk '{print $1}')"
    already="$(psql "$DB_URL" -tAc \
        "SELECT 1 FROM schema_migrations WHERE filename = '${base}' LIMIT 1")"
    if [ "$already" = "1" ]; then
        echo "  ✓ $base (already applied, skipping)"
        continue
    fi
    echo "  ▶ applying $base ..."
    psql "$DB_URL" -v ON_ERROR_STOP=1 -f "$file"
    psql "$DB_URL" -v ON_ERROR_STOP=1 -c \
        "INSERT INTO schema_migrations (filename, checksum) VALUES ('${base}', '${checksum}')"
    echo "  ✓ $base applied"
done

echo ""
echo "═════════════════════════════════════════"
echo "  All migrations applied. Current state:"
psql "$DB_URL" -c "SELECT filename, applied_at FROM schema_migrations ORDER BY applied_at"
