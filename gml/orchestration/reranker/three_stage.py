"""ThreeStageReranker — CrossEncoder → LLM → weighted-score.

Stage 1: ``CrossEncoderReranker`` (~50ms) narrows ~100 candidates to ~20.
Stage 2: ``LLMReranker`` (~1-2s) reads the 20 + question, ranks by
         direct-answer relevance. Specialized for hard categories
         (multi-hop, temporal, paraphrase).
Stage 3: ``ScoreReranker`` weights recency/authority/pin on top.

This is the highest-precision reranking stack — Tier 3 in the LOCOMO plan.
Reach for it when accuracy matters more than ~1-2s per query.
"""
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
from orchestration.reranker.llm_reranker import LLMReranker
from orchestration.reranker.score_reranker import ScoreReranker
from orchestration.sam._ollama_client import OllamaClient


class ThreeStageReranker(Reranker):
    """CE → LLM → score-weight pipeline.

    Args:
        config: project config (for ScoreReranker weights).
        llm_client: local-LLM client used by the LLM rerank stage.
        cross_encoder_model: model name for stage 1.
        ce_keep: how many to pass from CE to LLM. Default 20.
        llm_keep: how many to pass from LLM to ScoreReranker. Default 15.
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        llm_client: OllamaClient,
        cross_encoder_model: str = DEFAULT_CE_MODEL,
        ce_keep: int = 20,
        llm_keep: int = 15,
    ) -> None:
        self._cross = CrossEncoderReranker(model_name=cross_encoder_model)
        self._llm = LLMReranker(llm_client, max_candidates=ce_keep)
        self._score = ScoreReranker(config)
        self.ce_keep = ce_keep
        self.llm_keep = llm_keep

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        # Stage 1: Cross-encoder narrows to top-ce_keep
        ce_ranked = await self._cross.pick_best(hits, query, k=self.ce_keep)
        if not ce_ranked:
            return []

        # Stage 2: LLM reranker reads top-ce_keep, picks top-llm_keep
        # The LLM reranker takes RetrievalHit input; rebuild from RankedHit.
        ce_as_hits = [
            RetrievalHit(record=r.hit.record, similarity=r.semantic_score)
            for r in ce_ranked
        ]
        llm_ranked = await self._llm.pick_best(ce_as_hits, query, k=self.llm_keep)
        if not llm_ranked:
            llm_ranked = ce_ranked[: self.llm_keep]

        # Stage 3: ScoreReranker applies recency/authority on top of LLM order.
        # Use LLM's final_score (rank-derived) as the new similarity.
        promoted = [
            RetrievalHit(record=r.hit.record, similarity=r.final_score)
            for r in llm_ranked
        ]
        final = await self._score.pick_best(promoted, query, k=k)

        # Annotate with the chain of decisions
        ce_by_id = {r.hit.record.id: r for r in ce_ranked}
        llm_by_id = {r.hit.record.id: r for r in llm_ranked}
        annotated: list[RankedHit] = []
        for rh in final:
            rid = rh.hit.record.id
            note_parts = []
            if rid in ce_by_id:
                note_parts.append(f"ce={ce_by_id[rid].semantic_score:.2f}")
            if rid in llm_by_id:
                note_parts.append(f"llm={llm_by_id[rid].final_score:.2f}")
            note = " " + " ".join(note_parts) if note_parts else ""
            annotated.append(rh.model_copy(update={
                "score_reason": rh.score_reason + note,
            }))
        return annotated
