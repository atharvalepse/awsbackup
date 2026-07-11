#!/usr/bin/env python3
"""Review/apply pending entity merge candidates (migration 014).

Dry-run by default — prints what WOULD happen:

    python scripts/resolve_merge_candidates.py

Apply merges for strong candidates and reject hopeless ones:

    python scripts/resolve_merge_candidates.py --apply \
        --min-sim 0.60 --reject-below 0.50

Scope to one tenant with --user. DSN from GML_DATABASE_URL or --dsn.
Cron-able; exits non-zero on connection failure only.
"""
import argparse
import asyncio
import json
import os
import sys


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=os.environ.get("GML_DATABASE_URL"))
    ap.add_argument("--user", default=None, help="limit to one tenant")
    ap.add_argument("--min-sim", type=float, default=0.60,
                    help="auto-merge at/above this similarity (default 0.60)")
    ap.add_argument("--reject-below", type=float, default=None,
                    help="mark candidates below this similarity rejected")
    ap.add_argument("--apply", action="store_true",
                    help="actually merge/reject (default: dry-run report)")
    args = ap.parse_args()

    if not args.dsn:
        print("error: no DSN (set GML_DATABASE_URL or pass --dsn)", file=sys.stderr)
        return 2

    import asyncpg

    from orchestration.storage.entity_maintenance import resolve_merge_candidates

    pool = await asyncpg.create_pool(args.dsn, min_size=1, max_size=2)
    try:
        report = await resolve_merge_candidates(
            pool,
            user_id=args.user,
            min_sim=args.min_sim,
            reject_below=args.reject_below,
            apply=args.apply,
        )
    finally:
        await pool.close()

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"# entity merge candidates — {mode}, {len(report)} pending")
    for entry in report:
        print(json.dumps(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
