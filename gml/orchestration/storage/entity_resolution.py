"""Write-time entity resolution (migration 014).

Resolves a raw entity mention ("GML", "Gigzs Multi-LLM Layer", "Priya") to a
canonical per-tenant entity id, deterministically:

  1. exact alias        — normalized mention already known
  2. acronym            — initials of a multiword mention match an alias, or
                          a known multiword canonical's initials match the
                          mention ("gml" ⇄ "gigzs multi-llm layer")
  3. name containment   — single-token mention equals the first token of
                          exactly ONE known multiword alias (or vice versa):
                          "Priya" ⇄ "Priya Sharma". Ambiguity (two Priyas)
                          falls through — never guess.
  4. trigram fuzzy      — similarity >= STRONG (0.65): same entity, register
                          the new alias. GRAY zone (0.45–0.65): create a
                          PROVISIONAL entity + a reviewable merge candidate.
  5. new entity         — id = ent_<sha1(user_id, norm)[:12]>, so creation is
                          idempotent and order-independent (two concurrent
                          writers minting "GML" produce the same id).

All queries are tenant-scoped (explicit user_id; RLS backs it up). Callers
run this inside add_many's transaction, under the per-tenant advisory lock,
which is what makes gray-zone decisions deterministic under concurrency.
"""
from __future__ import annotations

import hashlib
import os
import re

STRONG_SIM = float(os.environ.get("GML_ER_STRONG_SIM", "0.65"))
GRAY_SIM = float(os.environ.get("GML_ER_GRAY_SIM", "0.45"))

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# Mentions too generic to resolve — storing them as entities just recreates
# the duplication problem under a different name.
_GENERIC = {"system", "project", "team", "company", "app", "service", "user", "it"}


def normalize(mention: str) -> str:
    m = _ARTICLE_RE.sub("", mention.strip())
    m = _WS_RE.sub(" ", m).strip(" .,:;!?\"'")
    return m.lower()


def acronym(norm: str) -> str | None:
    words = norm.split(" ")
    if len(words) < 2:
        return None
    return "".join(w[0] for w in words if w)


def _like_escape(s: str) -> str:
    """Escape LIKE wildcards in a mention ("100%_uptime service") so the
    containment queries match literally instead of as patterns."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def entity_id_for(user_id: str, norm: str) -> str:
    digest = hashlib.sha1(f"{user_id}\x1f{norm}".encode()).hexdigest()[:12]
    return f"ent_{digest}"


class EntityResolver:
    """Stateless; operates on the caller's open asyncpg connection."""

    async def resolve(self, conn, user_id: str, mention: str) -> str | None:
        """Resolve a mention to an entity id, creating entities/aliases as
        needed. Returns None for unresolvably generic mentions."""
        norm = normalize(mention)
        if not norm or norm in _GENERIC:
            return None

        # 1. exact alias
        row = await conn.fetchrow(
            "SELECT entity_id FROM entity_aliases WHERE user_id=$1 AND alias_norm=$2",
            user_id, norm,
        )
        if row:
            return row["entity_id"]

        # 2a. incoming multiword whose initials are a known alias
        acr = acronym(norm)
        if acr:
            row = await conn.fetchrow(
                "SELECT entity_id FROM entity_aliases WHERE user_id=$1 AND alias_norm=$2",
                user_id, acr,
            )
            if row:
                await self._add_alias(conn, user_id, norm, row["entity_id"], "acronym")
                return row["entity_id"]
        # 2b. incoming short mention that is the acronym of a known multiword
        # alias (initials computed in SQL over multiword aliases).
        if " " not in norm and 2 <= len(norm) <= 6:
            rows = await conn.fetch(
                """
                SELECT DISTINCT entity_id FROM entity_aliases
                WHERE user_id = $1 AND alias_norm LIKE '% %'
                  AND (SELECT string_agg(left(w, 1), '')
                       FROM unnest(string_to_array(alias_norm, ' ')) AS w) = $2
                """,
                user_id, norm,
            )
            if len(rows) == 1:
                await self._add_alias(conn, user_id, norm, rows[0]["entity_id"], "acronym")
                return rows[0]["entity_id"]

        # 3. first-token containment, only when unambiguous
        if " " not in norm:
            rows = await conn.fetch(
                "SELECT DISTINCT entity_id FROM entity_aliases "
                "WHERE user_id=$1 AND alias_norm LIKE $2 || ' %'",
                user_id, _like_escape(norm),
            )
            if len(rows) == 1:
                await self._add_alias(conn, user_id, norm, rows[0]["entity_id"], "write")
                return rows[0]["entity_id"]
        else:
            first = norm.split(" ", 1)[0]
            if first not in _GENERIC:
                rows = await conn.fetch(
                    "SELECT DISTINCT entity_id FROM entity_aliases "
                    "WHERE user_id=$1 AND alias_norm=$2",
                    user_id, first,
                )
                if len(rows) == 1:
                    # "Priya Sharma" arriving after "Priya" — but only if no
                    # other multiword alias starts with the same first token.
                    others = await conn.fetchval(
                        "SELECT count(DISTINCT entity_id) FROM entity_aliases "
                        "WHERE user_id=$1 AND alias_norm LIKE $2 || ' %'",
                        user_id, _like_escape(first),
                    )
                    if others == 0:
                        await self._add_alias(
                            conn, user_id, norm, rows[0]["entity_id"], "write"
                        )
                        return rows[0]["entity_id"]

        # 4. trigram fuzzy — the % operator (not similarity()>=x) so the
        # entity_aliases_trgm GIN index engages; a function-call predicate
        # forces a sequential scan over the tenant's whole alias table.
        # set_config(..., is_local=true) scopes the threshold to add_many's
        # transaction.
        await conn.execute(
            "SELECT set_config('pg_trgm.similarity_threshold', $1, true)",
            str(GRAY_SIM),
        )
        best = await conn.fetchrow(
            """
            SELECT entity_id, alias_norm, similarity(alias_norm, $2) AS sim
            FROM entity_aliases
            WHERE user_id = $1 AND alias_norm % $2
            ORDER BY sim DESC, alias_norm ASC
            LIMIT 1
            """,
            user_id, norm,
        )
        if best and best["sim"] >= STRONG_SIM:
            await self._add_alias(conn, user_id, norm, best["entity_id"], "fuzzy")
            return best["entity_id"]

        # 5. new entity (provisional when a gray-zone neighbour exists)
        eid = await self._create_entity(
            conn, user_id, norm, mention.strip(), provisional=bool(best)
        )
        if best:
            a, b = sorted((best["entity_id"], eid))
            await conn.execute(
                """
                INSERT INTO entity_merge_candidates
                    (user_id, entity_a, entity_b, similarity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, entity_a, entity_b) DO NOTHING
                """,
                user_id, a, b, float(best["sim"]),
            )
        return eid

    async def _create_entity(
        self, conn, user_id: str, norm: str, display: str, provisional: bool
    ) -> str:
        eid = entity_id_for(user_id, norm)
        await conn.execute(
            """
            INSERT INTO entities (id, user_id, canonical_name, provisional)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO NOTHING
            """,
            eid, user_id, display, provisional,
        )
        await self._add_alias(conn, user_id, norm, eid, "write")
        acr = acronym(norm)
        if acr and acr not in _GENERIC:
            await self._add_alias(conn, user_id, acr, eid, "acronym")
        return eid

    async def _add_alias(
        self, conn, user_id: str, alias_norm: str, entity_id: str, source: str
    ) -> None:
        await conn.execute(
            """
            INSERT INTO entity_aliases (user_id, alias_norm, entity_id, source)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, alias_norm) DO NOTHING
            """,
            user_id, alias_norm, entity_id, source,
        )

    async def canonical_name(self, conn, user_id: str, entity_id: str) -> str | None:
        return await conn.fetchval(
            "SELECT canonical_name FROM entities WHERE user_id=$1 AND id=$2",
            user_id, entity_id,
        )
