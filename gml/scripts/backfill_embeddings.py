#!/usr/bin/env python3
"""Backfill (and monitor) memories with a NULL embedding.

Such rows never match a vector query, so they're effectively invisible to
recall. This script finds them and re-embeds them with the configured embedder
(GML_EMBEDDER), and doubles as the monitoring check for a periodic job.

Usage:
    # one-off backfill
    python -m scripts.backfill_embeddings
    python -m scripts.backfill_embeddings --batch-size 128
    python -m scripts.backfill_embeddings --dry-run        # count + plan only

    # monitoring (cron): exits 1 if any NULL embeddings remain, 0 otherwise
    python -m scripts.backfill_embeddings --check

Requires GML_DATABASE_URL (Postgres backend). Runs with app.is_admin so it can
read/update across tenants.
"""
import argparse
import asyncio
import sys

from orchestration.server import _build_embedder_from_env
from orchestration.storage import get_pg_pool


async def _count_nulls(conn) -> int:
    async with conn.transaction():
        await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
        return await conn.fetchval(
            "SELECT count(*) FROM memories WHERE embedding IS NULL"
        )


async def main(batch_size: int, dry_run: bool, check: bool) -> int:
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        if check:
            n = await _count_nulls(conn)
            print(f"NULL embeddings: {n}")
            return 1 if n > 0 else 0

        async with conn.transaction():
            await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
            rows = await conn.fetch(
                "SELECT id, content FROM memories WHERE embedding IS NULL"
            )
        print(f"{len(rows)} rows with NULL embedding")
        if not rows or dry_run:
            return 0

        embedder = _build_embedder_from_env()
        done = skipped = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            vectors = await embedder.embed_batch([r["content"] or "" for r in batch])
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.is_admin', 'true', true)")
                for r, vec in zip(batch, vectors):
                    if not vec:
                        skipped += 1
                        print(f"  WARN empty vector for {r['id']} — skipped")
                        continue
                    await conn.execute(
                        "UPDATE memories SET embedding = $1 WHERE id = $2",
                        vec, r["id"],
                    )
                    done += 1
            print(f"  backfilled {min(i + batch_size, len(rows))}/{len(rows)}")
        print(f"done: {done} backfilled, {skipped} skipped (still NULL)")
        return 1 if skipped else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true", help="count + plan only")
    ap.add_argument(
        "--check", action="store_true",
        help="monitoring mode: exit 1 if any NULL embeddings remain",
    )
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.batch_size, args.dry_run, args.check)))
