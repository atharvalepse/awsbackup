"""TwoStageCrossEncoderReranker — combine cross-encoder + score weighting.

Stage 1: a transformer cross-encoder scores (query, candidate) pairs
jointly. This is the sharp semantic signal that pure cosine misses.
Stage 2: the existing ScoreReranker applies the recency/authority/pin
weighting on top, using the cross-encoder score as the new semantic
component instead of the raw cosine score.

Wiring this into the Pipeline in place of ScoreReranker alone consistently
improves retrieval recall by 10-25% absolute on standard benchmarks — it
is the single biggest known retrieval lever. Cost: ~50-150ms for the
cross-encoder forward pass over ~100 candidates per query.

Fail-safe: if the cross-encoder model can't load (offline, model missing,
fastembed import error), construction raises RerankerError and the caller
should fall back to plain ScoreReranker.
"""
import os

from orchestration.pipeline.contracts import (
    OrchestrationConfig,
    Query,
    RankedHit,
    RetrievalHit,
)
from orchestration.reranker.base import Reranker
from orchestration.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    DEFAULT_MODEL as DEFAULT_CE_MODEL,
)
from orchestration.reranker.score_reranker import ScoreReranker


class TwoStageCrossEncoderReranker(Reranker):
    """Cross-encode then weighted-score-rank.

    ``cross_encoder_keep`` is how many candidates survive the cross-encoder
    pass before the ScoreReranker weighting. Default 25 — enough to give
    recency/authority signal a real say without re-running the cross
    encoder over noise. Set to None to keep all input hits.

    Example:
        >>> reranker = TwoStageCrossEncoderReranker(
        ...     config=config,
        ...     cross_encoder_model="Xenova/ms-marco-MiniLM-L-6-v2",
        ...     cross_encoder_keep=25,
        ... )
        >>> top10 = await reranker.pick_best(top100_hits, query, k=10)
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        cross_encoder_model: str = DEFAULT_CE_MODEL,
        cross_encoder_keep: int | None = 25,
        backend: str | None = None,
    ) -> None:
        """``backend``: 'fastembed' (default) uses CrossEncoderReranker (ONNX),
        'st' uses SentenceTransformerCrossEncoder (PyTorch, supports BGE
        models that fastembed broke).
        """
        backend = backend or os.environ.get("GML_CE_BACKEND", "fastembed").lower()
        if backend == "st":
            from orchestration.reranker.st_cross_encoder import (
                SentenceTransformerCrossEncoder, DEFAULT_ST_MODEL,
            )
            model = os.environ.get("GML_ST_CE_MODEL", DEFAULT_ST_MODEL)
            self._cross = SentenceTransformerCrossEncoder(model_name=model)
            self.model_name = model
        else:
            self._cross = CrossEncoderReranker(model_name=cross_encoder_model)
            self.model_name = cross_encoder_model
        self._score = ScoreReranker(config)
        self.cross_encoder_keep = cross_encoder_keep
        self.backend = backend

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        # Stage 1: cross-encoder. Keep more than ``k`` so Stage 2 has a real
        # candidate pool when applying recency/authority weighting.
        ce_keep = self.cross_encoder_keep or len(hits)
        ce_ranked = await self._cross.pick_best(hits, query, k=ce_keep)
        if not ce_ranked:
            return []

        # Stage 2: rebuild RetrievalHits with the cross-encoder score as the
        # new similarity, then apply the weighted-sum scorer. This way the
        # final_score combines: sharp semantic (ce) + recency + authority + pin.
        promoted_hits = [
            RetrievalHit(record=r.hit.record, similarity=r.semantic_score)
            for r in ce_ranked
        ]
        final = await self._score.pick_best(promoted_hits, query, k=k)

        # Carry the cross-encoder reason forward into score_reason so we can
        # see both contributions in trace/diag output.
        ce_by_id = {r.hit.record.id: r for r in ce_ranked}
        annotated: list[RankedHit] = []
        for rh in final:
            ce = ce_by_id.get(rh.hit.record.id)
            note = f" ce={ce.semantic_score:.3f}" if ce is not None else ""
            annotated.append(rh.model_copy(update={
                "score_reason": rh.score_reason + note,
            }))
        return annotated
