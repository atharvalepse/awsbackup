"""EnsembleCrossEncoder — average scores from two cross-encoders.

Why ensemble: a single cross-encoder has consistent biases (training
data, architecture). Two different CEs make uncorrelated errors. Taking
the average score catches the cases where one CE confidently ranks the
wrong doc highest.

Typical setup:
  - FT'd bge-reranker-base   (LOCOMO-specialized, sharper margins)
  - jinaai/jina-reranker-v2   (multilingual, different training data)
Score = 0.5 × FT + 0.5 × jina

Expected lift on R@5: +2-4 points absolute. Modest but reliable.

Cost: 2× cross-encoder inference time (~3.5s/query instead of ~1.7s).
"""
import asyncio
import math
import os
from typing import Optional

from orchestration.errors import RerankerError
from orchestration.pipeline.contracts import Query, RankedHit, RetrievalHit
from orchestration.reranker.base import Reranker


class EnsembleCrossEncoder(Reranker):
    """Score-average two cross-encoders on the same candidates."""

    def __init__(
        self,
        primary: Reranker,
        secondary: Reranker,
        primary_weight: float = 0.5,
    ) -> None:
        """``primary`` and ``secondary`` are both Reranker-conformant
        (typically two CrossEncoderReranker / SentenceTransformerCrossEncoder
        instances). They MUST score over the same input list; we'll align
        outputs by ``RetrievalHit.record.id``.

        ``primary_weight`` ∈ [0, 1]: how much weight on the primary score.
        Defaults to 0.5 (equal blend).
        """
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError(f"primary_weight must be in [0, 1], got {primary_weight}")
        self.primary = primary
        self.secondary = secondary
        self.primary_weight = primary_weight

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        # Score both rerankers in parallel — they're CPU-bound but the
        # sentence-transformers threads aren't blocked by each other.
        # Take MORE than k from each so the merged set has enough candidates.
        pull_k = max(k * 4, 50)
        p_task = self.primary.pick_best(hits, query, k=pull_k)
        s_task = self.secondary.pick_best(hits, query, k=pull_k)
        p_ranked, s_ranked = await asyncio.gather(p_task, s_task)

        # Align by record id. If a hit didn't make either's top-pull_k,
        # treat its missing score as the lowest seen.
        p_by_id = {r.hit.record.id: r.semantic_score for r in p_ranked}
        s_by_id = {r.hit.record.id: r.semantic_score for r in s_ranked}
        p_min = min(p_by_id.values(), default=0.0)
        s_min = min(s_by_id.values(), default=0.0)

        # Build the merged candidate set
        all_ids = set(p_by_id) | set(s_by_id)
        combined: list[tuple[float, RetrievalHit, float, float]] = []
        for hit in hits:
            rid = hit.record.id
            if rid not in all_ids:
                continue
            p = p_by_id.get(rid, p_min)
            s = s_by_id.get(rid, s_min)
            blended = self.primary_weight * p + (1 - self.primary_weight) * s
            combined.append((blended, hit, p, s))

        combined.sort(key=lambda x: -x[0])
        top = combined[:k]

        out: list[RankedHit] = []
        for blended, hit, p, s in top:
            out.append(RankedHit(
                hit=hit,
                semantic_score=blended,
                recency_score=0.0,
                authority_score=hit.record.authority_score,
                pin_boost=1.0 if hit.record.pinned else 0.0,
                final_score=blended,
                score_reason=f"ensemble: {self.primary_weight:.1f}*{p:.3f} + "
                             f"{1-self.primary_weight:.1f}*{s:.3f} = {blended:.3f}",
            ))
        return out


def make_ensemble_from_env() -> Optional[EnsembleCrossEncoder]:
    """Build an EnsembleCrossEncoder from GML_CE_ENSEMBLE env if set.

    Format: GML_CE_ENSEMBLE="primary_path|secondary_model|weight"
    Example: GML_CE_ENSEMBLE="models/ce_locomo_ft|jinaai/jina-reranker-v2-base-multilingual|0.7"
    Returns None when env isn't set.
    """
    val = os.environ.get("GML_CE_ENSEMBLE", "").strip()
    if not val:
        return None
    parts = val.split("|")
    if len(parts) < 2:
        return None
    primary_path = parts[0].strip()
    secondary_model = parts[1].strip()
    weight = float(parts[2]) if len(parts) > 2 else 0.5

    # Build primary (sentence-transformers if local path, else fastembed)
    from orchestration.reranker.cross_encoder_reranker import CrossEncoderReranker
    from orchestration.reranker.st_cross_encoder import SentenceTransformerCrossEncoder
    if os.path.exists(primary_path):
        primary: Reranker = SentenceTransformerCrossEncoder(model_name=primary_path)
    else:
        primary = CrossEncoderReranker(model_name=primary_path)

    if "/" in secondary_model and os.path.exists(secondary_model):
        secondary: Reranker = SentenceTransformerCrossEncoder(model_name=secondary_model)
    else:
        secondary = CrossEncoderReranker(model_name=secondary_model)

    return EnsembleCrossEncoder(primary, secondary, primary_weight=weight)
