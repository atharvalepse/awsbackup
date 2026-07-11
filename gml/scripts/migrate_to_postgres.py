"""Migrate JSONL stores → Postgres + pgvector. Idempotent, parallel-safe.

What it migrates
----------------
  ~/.gml/users.jsonl       →  users + user_keys
  ~/.gml/memories.jsonl    →  memories  (computes embeddings on the fly)

Paths are overridable via env vars (GML_USER_KEYS_FILE,
GML_MEMORY_STORE_PATH) — same as the production code reads.

Idempotency: every INSERT uses `ON CONFLICT (...) DO NOTHING`, so running
this script twice is safe. The source JSONL files are NEVER modified —
this is a copy operation. To re-migrate from scratch, TRUNCATE the
Postgres tables first.

Usage
-----
    .venv/bin/python scripts/migrate_to_postgres.py \\
        --database-url "postgresql://gml_app:PASS@127.0.0.1/gml" \\
        --user-id atharva          # which user owns the memories.jsonl content
        [--dry-run]
        [--batch-size 100]

Why we need `--user-id`: the existing memories.jsonl is single-tenant
(everything was implicitly "my memories"). In Postgres every row has a
user_id. This flag says "assign all unassigned memories to user X".

Requires `asyncpg` and `pgvector`:
    pip install asyncpg "pgvector[asyncpg]"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import asyncpg
except ImportError:
    sys.exit("Missing dep: pip install asyncpg")

try:
    from pgvector.asyncpg import register_vector
except ImportError:
    sys.exit("Missing dep: pip install pgvector")


DEFAULT_USER_KEYS_PATH = Path.home() / ".gml" / "users.jsonl"
DEFAULT_MEMORIES_PATH = Path.home() / ".gml" / "memories.jsonl"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [warn] skipped malformed line in {path.name}: {exc}", file=sys.stderr)


def _chunked(it: Iterable, n: int) -> Iterable[list]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _parse_ts(value) -> datetime:
    """Tolerantly parse a timestamp from JSONL. Defaults to now() on failure."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _byte_size(memory: dict) -> int:
    """Approximate the bytes a memory consumes — used for quota tracking."""
    content = memory.get("content", "") or ""
    extras = " ".join(
        str(memory.get(k) or "") for k in ("entity", "attribute", "value", "summary_short")
    )
    return len(content.encode("utf-8")) + len(extras.encode("utf-8"))


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


async def ensure_user(conn: asyncpg.Connection, user_id: str) -> None:
    """Create the user row if missing. Quota defaults to 1 GiB."""
    await conn.execute(
        """
        INSERT INTO users (user_id, plan, quota_bytes)
        VALUES ($1, 'free', 1073741824)
        ON CONFLICT (user_id) DO NOTHING
        """,
        user_id,
    )


async def migrate_user_keys(conn: asyncpg.Connection, path: Path, dry_run: bool) -> tuple[int, int]:
    n_users, n_keys = 0, 0
    seen_users: set[str] = set()
    for rec in _iter_jsonl(path):
        user_id = rec.get("user_id")
        key = rec.get("key")
        if not user_id or not key:
            continue
        if user_id not in seen_users:
            if not dry_run:
                await ensure_user(conn, user_id)
            seen_users.add(user_id)
            n_users += 1
        if dry_run:
            continue
        result = await conn.execute(
            """
            INSERT INTO user_keys (key, user_id, label, created_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (key) DO NOTHING
            """,
            key, user_id, rec.get("label"), _parse_ts(rec.get("created_at")),
        )
        if result.endswith(" 1"):
            n_keys += 1
    return n_users, n_keys


async def migrate_memories(
    conn: asyncpg.Connection,
    path: Path,
    user_id: str,
    embedder,
    *,
    batch_size: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Batched insert. Computes embeddings on the fly for any memory whose
    raw_metadata doesn't already carry one."""
    if not path.exists():
        return 0, 0

    # The user must exist before we can FK-link memories to it.
    if not dry_run:
        await ensure_user(conn, user_id)

    total_seen = 0
    total_inserted = 0

    for batch in _chunked(_iter_jsonl(path), batch_size):
        # Compute embeddings for the whole batch in one pass — much faster
        # than one-at-a-time, even on FastEmbed (ONNX amortizes overhead).
        contents = [m.get("content", "") or "" for m in batch]
        vectors = await embedder.embed_batch(contents)

        rows: list[tuple] = []
        for mem, vec in zip(batch, vectors):
            total_seen += 1
            mid = mem.get("id") or f"migrated-{int(time.time()*1000)}-{total_seen}"
            ts = _parse_ts(mem.get("timestamp"))
            rows.append((
                mid,
                user_id,
                mem.get("content", "") or "",
                mem.get("entity"),
                mem.get("attribute"),
                mem.get("value"),
                mem.get("source", "migrated"),
                float(mem.get("authority_score", 0.7)),
                bool(mem.get("pinned", False)),
                ts,
                json.dumps(mem.get("raw_metadata") or {}),
                mem.get("summary_short"),
                vec,
                _byte_size(mem),
            ))

        if dry_run:
            continue

        # executemany doesn't return per-row affected counts, so we use a
        # single SQL VALUES list with returning to know how many landed.
        # Bounded VALUES generation is fine — batch_size is small (≤200).
        # Wrap in a transaction so a single bad row doesn't poison the batch.
        async with conn.transaction():
            inserted = await conn.fetchval(
                """
                WITH ins AS (
                    INSERT INTO memories (
                        id, user_id, content, entity, attribute, value,
                        source, authority_score, pinned, "timestamp",
                        raw_metadata, summary_short, embedding, byte_size
                    )
                    SELECT * FROM UNNEST(
                        $1::text[], $2::text[], $3::text[], $4::text[], $5::text[],
                        $6::text[], $7::text[], $8::real[], $9::boolean[],
                        $10::timestamptz[], $11::jsonb[], $12::text[],
                        $13::vector(384)[], $14::int[]
                    )
                    ON CONFLICT (id) DO NOTHING
                    RETURNING 1
                )
                SELECT count(*) FROM ins
                """,
                [r[0] for r in rows],   # id
                [r[1] for r in rows],   # user_id
                [r[2] for r in rows],   # content
                [r[3] for r in rows],   # entity
                [r[4] for r in rows],   # attribute
                [r[5] for r in rows],   # value
                [r[6] for r in rows],   # source
                [r[7] for r in rows],   # authority_score
                [r[8] for r in rows],   # pinned
                [r[9] for r in rows],   # timestamp
                [r[10] for r in rows],  # raw_metadata
                [r[11] for r in rows],  # summary_short
                [r[12] for r in rows],  # embedding
                [r[13] for r in rows],  # byte_size
            )
        total_inserted += int(inserted or 0)
        print(f"  batch: {len(batch)} seen, {inserted} inserted (total {total_inserted}/{total_seen})")

    return total_seen, total_inserted


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("GML_DATABASE_URL"))
    parser.add_argument("--user-id", required=True,
                        help="Owner for migrated memories (creates the row if missing)")
    parser.add_argument("--users-jsonl", default=str(DEFAULT_USER_KEYS_PATH))
    parser.add_argument("--memories-jsonl", default=str(DEFAULT_MEMORIES_PATH))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Read + count but don't write to the DB")
    args = parser.parse_args(argv)

    if not args.database_url:
        sys.exit("Provide --database-url or set GML_DATABASE_URL")

    print(f"► Connecting to {args.database_url.split('@')[-1]}")
    conn = await asyncpg.connect(args.database_url)
    try:
        # pgvector type registration — lets us pass Python lists / numpy
        # arrays as `vector(384)` columns.
        await register_vector(conn)

        # User keys
        print(f"► Migrating user keys from {args.users_jsonl}")
        n_users, n_keys = await migrate_user_keys(
            conn, Path(args.users_jsonl), args.dry_run
        )
        print(f"  users seen={n_users}, keys inserted={n_keys}")

        # Memories
        print(f"► Migrating memories from {args.memories_jsonl} → user_id={args.user_id!r}")
        from orchestration.embedder import FastEmbedEmbedder
        embedder = FastEmbedEmbedder()
        seen, inserted = await migrate_memories(
            conn, Path(args.memories_jsonl), args.user_id, embedder,
            batch_size=args.batch_size, dry_run=args.dry_run,
        )
        print(f"  memories: seen={seen}, inserted={inserted}, skipped={seen - inserted}")

        # Final byte-usage report — the trigger maintains this automatically.
        if not args.dry_run:
            row = await conn.fetchrow(
                "SELECT bytes_used, quota_bytes FROM users WHERE user_id = $1",
                args.user_id,
            )
            if row:
                pct = (row["bytes_used"] / row["quota_bytes"]) * 100 if row["quota_bytes"] else 0
                print(f"► Post-migration quota: {row['bytes_used']:,} / {row['quota_bytes']:,} bytes ({pct:.1f}%)")
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
