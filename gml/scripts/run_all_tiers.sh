#!/bin/bash
# Three-tier LOCOMO comparison run.
#
# Same subset, same conversations, same QAs across tiers. The only thing
# that changes between runs is which feature env vars are on.
#
# Tier 1:  cross-encoder (A1) + bge-large (A3) + top-100 (A4) + tightened
#          SAM-skip (A2) + sliding-windows (B2, free) + MinHash dedup (B5, free)
# Tier 2:  Tier 1 + HyDE (B3) + entity index (B1) + AAL tuples (B4)
# Tier 3:  Tier 2 + LLM reranker (C3) + query router (C1, always on)
#
# Results are written to /tmp/locomo-t{1,2,3}.json so they can be compared
# after the run finishes.

set -u
cd /Users/atharvalepse/Projects/gml-orchestration

DATA=/tmp/locomo/locomo10.json
LIMIT=${LIMIT:-1}                 # conversations
QAS=${QAS:-50}                    # QAs per conversation

COMMON_FLAGS=(
  --data "$DATA"
  --limit "$LIMIT"
  --max-qa-per-conv "$QAS"
)

run_tier() {
  local name=$1
  local cp="/tmp/locomo-${name}.json"
  local log="/tmp/locomo-${name}.log"
  echo ""
  echo "==================================================================="
  echo "  $name  starting at $(date '+%H:%M:%S')"
  echo "  log: $log"
  echo "  checkpoint: $cp"
  echo "  env: GML_EMBED_MODEL=$GML_EMBED_MODEL"
  echo "       GML_CROSS_ENCODER=$GML_CROSS_ENCODER"
  echo "       GML_HYDE=$GML_HYDE  GML_ENTITY_INDEX=$GML_ENTITY_INDEX"
  echo "       GML_AAL_TUPLES=$GML_AAL_TUPLES  GML_LLM_RERANKER=$GML_LLM_RERANKER"
  echo "       ingest=$INGEST_MODE"
  echo "==================================================================="
  rm -f "$cp" "$log"
  .venv/bin/python scripts/benchmark_locomo.py \
    "${COMMON_FLAGS[@]}" --ingest-mode "$INGEST_MODE" \
    --checkpoint "$cp" > "$log" 2>&1
  local exit_code=$?
  echo "  $name finished at $(date '+%H:%M:%S')  exit=$exit_code"
  return $exit_code
}

# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------
export GML_EMBED_MODEL=BAAI/bge-large-en-v1.5
export GML_CROSS_ENCODER=1
export GML_HYDE=0
export GML_ENTITY_INDEX=0
export GML_AAL_TUPLES=0
export GML_LLM_RERANKER=0
export INGEST_MODE=raw
run_tier T1

# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
export GML_HYDE=1
export GML_ENTITY_INDEX=1
export GML_AAL_TUPLES=1
export INGEST_MODE=raw+aal
run_tier T2

# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------
export GML_LLM_RERANKER=1
run_tier T3

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  THREE-TIER COMPARISON"
echo "==================================================================="
.venv/bin/python - <<'PY'
import json
from collections import defaultdict

rows = []
for tier in ("T1", "T2", "T3"):
    path = f"/tmp/locomo-{tier}.json"
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        rows.append((tier, None))
        continue
    runs = d.get("runs", [])
    if not runs:
        rows.append((tier, None))
        continue
    cat = defaultdict(list)
    total_qa_ms = 0
    total_ingest_ms = 0
    total_mem = 0
    for r in runs:
        total_qa_ms += r.get("qa_ms", 0)
        total_ingest_ms += r.get("ingest_ms", 0)
        total_mem += r.get("n_mem", 0)
        for qa in r["qa_results"]:
            cat[qa["category"]].append(qa["recall"])
    n_qa = sum(len(v) for v in cat.values())
    overall = sum(v for vs in cat.values() for v in vs) / max(n_qa, 1)
    rows.append((tier, {
        "n_mem": total_mem, "n_qa": n_qa,
        "ingest_s": total_ingest_ms / 1000.0,
        "qa_s": total_qa_ms / 1000.0,
        "overall": overall,
        "per_cat": {c: (sum(v) / len(v), len(v)) for c, v in cat.items()},
    }))

print(f"{'tier':<6} {'mem':>5} {'qa':>4} {'ingest_s':>9} {'qa_s':>7} {'recall':>8}  per-cat (avg recall, n)")
print("-" * 100)
for tier, r in rows:
    if r is None:
        print(f"{tier:<6} (no results)")
        continue
    per_cat = "  ".join(
        f"c{c}:{avg:.2f}({n})" for c, (avg, n) in sorted(r["per_cat"].items())
    )
    print(
        f"{tier:<6} {r['n_mem']:>5} {r['n_qa']:>4} {r['ingest_s']:>9.1f} "
        f"{r['qa_s']:>7.1f} {r['overall']:>8.3f}  {per_cat}"
    )

# Marginal lifts
if all(r[1] is not None for r in rows[:2]):
    t1, t2 = rows[0][1]["overall"], rows[1][1]["overall"]
    print(f"\nMarginal lift T1→T2: {t2-t1:+.3f} ({(t2-t1)/max(t1,0.001)*100:+.1f}%)")
if all(r[1] is not None for r in rows[1:]):
    t2, t3 = rows[1][1]["overall"], rows[2][1]["overall"]
    print(f"Marginal lift T2→T3: {t3-t2:+.3f} ({(t3-t2)/max(t2,0.001)*100:+.1f}%)")
PY
echo "==================================================================="
