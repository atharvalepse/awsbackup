import os
import sys

from orchestration.pipeline.contracts import OrchestrationConfig
from orchestration.reranker.base import Reranker
from orchestration.reranker.cross_encoder_reranker import CrossEncoderReranker
from orchestration.reranker.ensemble import EnsembleCrossEncoder
from orchestration.reranker.llm_reranker import LLMReranker
from orchestration.reranker.negation_aware import NegationAwareReranker
from orchestration.reranker.score_reranker import ScoreReranker
from orchestration.reranker.three_stage import ThreeStageReranker
from orchestration.reranker.two_stage import TwoStageCrossEncoderReranker

# Optional sentence-transformers-backed cross-encoder. May not be installed.
try:
    from orchestration.reranker.st_cross_encoder import SentenceTransformerCrossEncoder
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformerCrossEncoder = None  # type: ignore
    _ST_AVAILABLE = False


def make_reranker(config: OrchestrationConfig) -> Reranker:
    """Pick the best reranker available, with graceful fallback.

    Selection order (highest precision first):
      1. ``ThreeStageReranker`` (cross-encoder → LLM → score-weighted) when
         ``GML_LLM_RERANKER=1`` (off by default — adds ~1-2s per query).
      2. ``TwoStageCrossEncoderReranker`` (cross-encoder + score-weighted)
         when ``GML_CROSS_ENCODER=1`` (on by default).
      3. ``ScoreReranker`` (plain weighted-sum) — last-resort fallback.

    Each tier falls back to the next if init fails (model missing, LLM
    unreachable, etc.) so the pipeline always boots.
    """
    # Tier 3: LLM reranker
    if os.environ.get("GML_LLM_RERANKER", "0") == "1":
        try:
            from orchestration.sam._ollama_client import make_local_llm_client
            llm_client = make_local_llm_client()
            rr = ThreeStageReranker(config, llm_client=llm_client)
            sys.stderr.write("• Reranker: ThreeStage (CE → LLM → score-weighted)\n")
            return rr
        except Exception as exc:
            sys.stderr.write(
                f"⚠ ThreeStageReranker unavailable ({type(exc).__name__}: {exc}); "
                "falling back to TwoStage.\n"
            )

    # Tier 2: Cross-encoder + score-weighted
    if os.environ.get("GML_CROSS_ENCODER", "1") == "0":
        sys.stderr.write("• Reranker: ScoreReranker (cross-encoder disabled via GML_CROSS_ENCODER=0)\n")
        return ScoreReranker(config)

    try:
        # Optional ensemble path: GML_CE_ENSEMBLE=primary|secondary|weight
        # When set, blends two CE models on the same candidate set. Skips the
        # TwoStageCrossEncoderReranker (which already does CE re-scoring of
        # the score-weighted top-K) because the ensemble IS the CE stage.
        from orchestration.reranker.ensemble import make_ensemble_from_env
        ensemble = make_ensemble_from_env()
        if ensemble is not None:
            sys.stderr.write(
                f"• Reranker: EnsembleCrossEncoder "
                f"(weight={ensemble.primary_weight})\n"
            )
            rr: Reranker = ensemble
        else:
            rr = TwoStageCrossEncoderReranker(config)
            sys.stderr.write(f"• Reranker: TwoStage (cross-encoder: {rr.model_name})\n")
        # Optional negation-aware wrapper (default ON: catches "we don't use X")
        if os.environ.get("GML_NEGATION_AWARE", "1") == "1":
            rr = NegationAwareReranker(rr)
            sys.stderr.write("• Reranker: + NegationAware wrapper\n")
        # Optional temporal-aware wrapper (POST-CE date boost — fixes the
        # weak temporal F1 category by promoting date-bearing memories AFTER
        # the cross-encoder has scored them). Gated by GML_TEMPORAL_AWARE
        # (default OFF to preserve baseline behavior; turn on for LOCOMO
        # temporal-heavy benches).
        if os.environ.get("GML_TEMPORAL_AWARE", "0") == "1":
            from orchestration.reranker.temporal_aware import TemporalAwareReranker
            rr = TemporalAwareReranker(rr)
            sys.stderr.write(
                f"• Reranker: + TemporalAware wrapper (boost={rr.boost})\n"
            )
        return rr
    except Exception as exc:
        sys.stderr.write(
            f"⚠ Cross-encoder unavailable ({type(exc).__name__}: {exc}); "
            "falling back to ScoreReranker.\n"
        )
        return ScoreReranker(config)


__all__ = [
    "CrossEncoderReranker", "EnsembleCrossEncoder", "LLMReranker",
    "NegationAwareReranker", "Reranker", "ScoreReranker", "ThreeStageReranker",
    "TwoStageCrossEncoderReranker", "make_reranker",
]
