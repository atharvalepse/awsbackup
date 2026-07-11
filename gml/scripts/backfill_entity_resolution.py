"""Backfill entity resolution over existing memories (migration 014).

For every memory with an entity mention and no entity_id, resolve the mention
through the same EntityResolver the write path uses (so backfill and live
writes can never disagree), and set memories.entity_id.

Per-tenant, batched, and idempotent: re-running skips already-resolved rows.
Runs under the same per-tenant advisory lock as live writes so a backfill
racing a live ingest can't make divergent gray-zone decisions.

Usage:
    GML_DATABASE_URL=postgresql://... python scripts/backfill_entity_resolution.py
    # or pass the DSN:
    python scripts/backfill_entity_resolution.py --dsn postgresql://...
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.storage.entity_resolution import EntityResolver  # noqa: E402

BATCH = 500


async def backfill_user(pool: asyncpg.Pool, user_id: str) -> tuple[int, int]:
    resolver = EntityResolver()
    resolved = skipped = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))", user_id
            )
            rows = await conn.fetch(
                """
                SELECT id, entity FROM memories
                WHERE user_id = $1 AND entity IS NOT NULL AND entity <> ''
                  AND entity_id IS NULL
                ORDER BY "timestamp" ASC
                """,
                user_id,
            )
            for r in rows:
                eid = await resolver.resolve(conn, user_id, r["entity"])
                if eid is None:
                    skipped += 1
                    continue
                await conn.execute(
                    "UPDATE memories SET entity_id = $2 WHERE id = $1", r["id"], eid
                )
                resolved += 1
    return resolved, skipped


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("GML_DATABASE_URL"))
    args = ap.parse_args()
    if not args.dsn:
        print("set GML_DATABASE_URL or pass --dsn", file=sys.stderr)
        return 2

    pool = await asyncpg.create_pool(args.dsn, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            # session-level (3rd arg false): outside a txn a local set_config
            # expires immediately and RLS would hide every row
            await conn.execute("SELECT set_config('app.is_admin', 'true', false)")
            users = [
                r["user_id"] for r in await conn.fetch(
                    "SELECT DISTINCT user_id FROM memories "
                    "WHERE entity IS NOT NULL AND entity <> '' AND entity_id IS NULL"
                )
            ]
        total_r = total_s = 0
        for uid in users:
            r, s = await backfill_user(pool, uid)
            total_r += r
            total_s += s
            print(f"  {uid}: resolved={r} skipped(generic)={s}")
        async with pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.is_admin', 'true', false)")
            n_ent = await conn.fetchval("SELECT count(*) FROM entities")
            n_alias = await conn.fetchval("SELECT count(*) FROM entity_aliases")
            n_merge = await conn.fetchval(
                "SELECT count(*) FROM entity_merge_candidates WHERE status='pending'"
            )
        print(f"\nbackfill done: {total_r} resolved, {total_s} generic-skipped "
              f"across {len(users)} tenant(s); {n_ent} entities, {n_alias} aliases, "
              f"{n_merge} pending merge candidate(s)")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
