# Database migrations

Numbered SQL files, applied in order. Each migration is idempotent (`CREATE … IF NOT EXISTS`, `INSERT … ON CONFLICT DO NOTHING`) so re-running on an already-migrated DB is a no-op.

## How to apply

### Option A: one at a time (when you want to review what changes before/after)

```bash
psql "$GML_DATABASE_URL" -f migrations/001_extensions.sql
psql "$GML_DATABASE_URL" -f migrations/002_users_and_keys.sql
psql "$GML_DATABASE_URL" -f migrations/003_memories.sql
psql "$GML_DATABASE_URL" -f migrations/004_byte_tracking.sql
psql "$GML_DATABASE_URL" -f migrations/005_row_level_security.sql
psql "$GML_DATABASE_URL" -f migrations/006_fts.sql
psql "$GML_DATABASE_URL" -f migrations/007_views.sql
```

### Option B: all at once (clean install)

```bash
bash migrations/apply_all.sh "$GML_DATABASE_URL"
```

### Option C: which migrations are applied? (for production change tracking)

```sql
SELECT * FROM schema_migrations ORDER BY applied_at;
```

`apply_all.sh` records every applied file in a `schema_migrations` table; if you re-run, already-applied files are skipped.

## Platform notes

| Platform | Notes |
|---|---|
| **Self-hosted Postgres** (our recommended GCP setup) | Apply as the `postgres` superuser. `deploy/gcp/02-install-postgres.sh` runs `001_extensions.sql` already. |
| **Cloud SQL Postgres** | The `postgres` user is reserved — apply migrations as `cloudsqlsuperuser`. Extensions: enable `vector`, `pg_trgm`, `pgcrypto` in the Cloud SQL console *before* `001_extensions.sql` runs (it'll error otherwise). |
| **Local dev (docker)** | `docker run --rm -e POSTGRES_PASSWORD=dev pgvector/pgvector:pg16` gives you a pgvector-ready image; same migrations apply. |

## What each migration does

| File | Purpose | Reversible? |
|---|---|---|
| 001 | Enable extensions: `vector`, `pg_trgm`, `pgcrypto` | No (extension creation; data-safe to drop manually) |
| 002 | `users` + `user_keys` tables | Drop tables to reverse |
| 003 | `memories` table + non-FTS indexes | Drop table to reverse |
| 004 | Byte-tracking trigger + function | Drop trigger + function |
| 005 | Row-level security policies | `DISABLE ROW LEVEL SECURITY` per table |
| 006 | `content_tsv` generated column + GIN index | `ALTER TABLE … DROP COLUMN` |
| 007 | `user_quota_status` view | `DROP VIEW` |

## Rolling back

Each migration adds the matching down-migration as a SQL comment at the top. Postgres has no "automatic rollback" — to undo, run the down-migration script manually:

```bash
psql "$GML_DATABASE_URL" -f migrations/006_fts.sql.down.sql
```

(Down-migration files are NOT yet present — we'll add them once any migration has reached production. Adding down-migration before any production deploy is over-engineering.)
