"""Entity merge-candidate maintenance (migration 014's missing consumer).

Gray-zone fuzzy matches create PROVISIONAL entities plus rows in
entity_merge_candidates — which, pre-this-module, nothing ever read,
resolved, or expired, so duplicate entities accumulated silently.

merge_entities() is the primitive: repoint aliases and memories from the
losing entity to the surviving one, then delete the loser (FK cascades
clean its candidate rows). resolve_merge_candidates() walks the pending
queue per tenant and applies policy: auto-merge at/above ``min_sim``,
optionally reject below ``reject_below``, dry-run by default.

Run via scripts/resolve_merge_candidates.py (cron-able) or call
resolve_merge_candidates() from an admin endpoint.
"""
from __future__ import annotations

from orchestration.observability.logging import StructuredLogger

slog = StructuredLogger("storage.entity_maintenance")


async def merge_entities(conn, user_id: str, loser: str, survivor: str) -> dict:
    """Fold ``loser`` into ``survivor`` for one tenant. Caller owns the
    transaction; runs under the tenant advisory lock so concurrent ingests
    can't resolve against a half-merged vocabulary."""
    if loser == survivor:
        raise ValueError("loser and survivor must differ")
    aliases = await conn.execute(
        "UPDATE entity_aliases SET entity_id = $3 "
        "WHERE user_id = $1 AND entity_id = $2",
        user_id, loser, survivor,
    )
    memories = await conn.execute(
        "UPDATE memories SET entity_id = $3 "
        "WHERE user_id = $1 AND entity_id = $2",
        user_id, loser, survivor,
    )
    # Cascade removes the loser's candidate rows (entity_a/b FK).
    await conn.execute(
        "DELETE FROM entities WHERE user_id = $1 AND id = $2", user_id, loser
    )
    counts = {
        "aliases_repointed": int(aliases.split()[-1]),
        "memories_repointed": int(memories.split()[-1]),
    }
    slog.info(
        event="entities_merged", user_id=user_id,
        loser=loser, survivor=survivor, **counts,
    )
    return counts


def _pick_survivor(row) -> tuple[str, str]:
    """(survivor, loser): a non-provisional entity always survives over a
    provisional one; ties broken by creation order (older survives, so ids
    referenced longest stay stable)."""
    a = (row["entity_a"], row["a_provisional"], row["a_created"])
    b = (row["entity_b"], row["b_provisional"], row["b_created"])
    for x, y in ((a, b), (b, a)):
        if not x[1] and y[1]:
            return x[0], y[0]
    return (a[0], b[0]) if a[2] <= b[2] else (b[0], a[0])


async def resolve_merge_candidates(
    pool,
    user_id: str | None = None,
    min_sim: float = 0.60,
    reject_below: float | None = None,
    apply: bool = False,
) -> list[dict]:
    """Walk pending merge candidates and apply policy. Returns a report of
    one dict per candidate with the action taken (or planned, in dry-run)."""
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.is_admin', 'true', false)")
        rows = await conn.fetch(
            """
            SELECT c.user_id, c.entity_a, c.entity_b, c.similarity,
                   ea.canonical_name AS a_name, ea.provisional AS a_provisional,
                   ea.created_at AS a_created,
                   eb.canonical_name AS b_name, eb.provisional AS b_provisional,
                   eb.created_at AS b_created
            FROM entity_merge_candidates c
            JOIN entities ea ON ea.id = c.entity_a AND ea.user_id = c.user_id
            JOIN entities eb ON eb.id = c.entity_b AND eb.user_id = c.user_id
            WHERE c.status = 'pending'
              AND ($1::text IS NULL OR c.user_id = $1)
            ORDER BY c.user_id, c.similarity DESC
            """,
            user_id,
        )

        report: list[dict] = []
        for row in rows:
            sim = float(row["similarity"] or 0.0)
            entry = {
                "user_id": row["user_id"],
                "entity_a": f"{row['entity_a']} ({row['a_name']})",
                "entity_b": f"{row['entity_b']} ({row['b_name']})",
                "similarity": round(sim, 4),
            }
            if sim >= min_sim:
                survivor, loser = _pick_survivor(row)
                entry["action"] = f"merge {loser} -> {survivor}"
                if apply:
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT pg_advisory_xact_lock(hashtext($1))",
                            row["user_id"],
                        )
                        entry.update(await merge_entities(
                            conn, row["user_id"], loser, survivor
                        ))
            elif reject_below is not None and sim < reject_below:
                entry["action"] = "reject"
                if apply:
                    await conn.execute(
                        "UPDATE entity_merge_candidates SET status = 'rejected' "
                        "WHERE user_id = $1 AND entity_a = $2 AND entity_b = $3",
                        row["user_id"], row["entity_a"], row["entity_b"],
                    )
            else:
                entry["action"] = "keep_pending"
            entry["applied"] = bool(apply and entry["action"] != "keep_pending")
            report.append(entry)
        return report
