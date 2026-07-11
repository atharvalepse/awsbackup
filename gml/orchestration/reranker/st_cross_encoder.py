"""SentenceTransformerCrossEncoder — cross-encoder via sentence-transformers.

Why this exists alongside ``CrossEncoderReranker`` (the fastembed-backed
one): fastembed's curated ONNX list is limited and `bge-reranker-base`
has a broken ONNX file in current fastembed versions. The canonical
HuggingFace BGE family is the strongest open cross-encoder line, and
sentence-transformers loads them directly without the fastembed shim.

Defaults to ``BAAI/bge-reranker-base`` (smaller cousin of the L-large
flagship). On CPU: ~17ms per (query, doc) pair → ~1.7s for top-100
reranking. The full ``bge-reranker-large`` (~50ms per pair, ~5s for
top-100) is also supported but generally too slow for the bench unless
top-K is capped at 30.

Same interface as the existing :class:`CrossEncoderReranker` so it's a
drop-in replacement.
"""
import asyncio
import math
import os

from orchestration.errors import RerankerError
from orchestration.pipeline.contracts import Query, RankedHit, RetrievalHit
from orchestration.reranker.base import Reranker
from orchestration.reranker.pair_cache import PairScoreCache


# Default model. The chain:
#   BAAI/bge-reranker-base   (~1GB, 17ms/pair on CPU, ~+8-12% BEIR vs MiniLM)  ← default
#   BAAI/bge-reranker-large  (~1.3GB, ~50ms/pair on CPU, sharpest but slow)
# Override with GML_ST_CE_MODEL env var.
DEFAULT_ST_MODEL = os.environ.get("GML_ST_CE_MODEL", "BAAI/bge-reranker-base")


def _build_doc_text(hit: RetrievalHit) -> str:
    """Same shape the fastembed CE uses — content + optional entity/attr/value."""
    parts = [hit.record.content]
    rec = hit.record
    if rec.entity:
        prefix = rec.entity
        if rec.attribute:
            prefix += f": {rec.attribute}"
        if rec.value:
            prefix += f" = {rec.value}"
        parts.append(f"[{prefix}]")
    return " ".join(parts)


class SentenceTransformerCrossEncoder(Reranker):
    """Cross-encoder reranker backed by sentence-transformers (not fastembed).

    Loads BAAI/bge-reranker-* models directly from HuggingFace, which is the
    real strongest-class open cross-encoder. The model is loaded once at
    construction and stays in RAM.

    Returns :class:`RankedHit` objects with the cross-encoder score in
    ``semantic_score`` (sigmoid-normalized to [0, 1]) and ``final_score``
    set to the same — the downstream :class:`ScoreReranker` adds the
    recency/authority/pin weighting on top.
    """

    def __init__(self, model_name: str = DEFAULT_ST_MODEL, device: str = "cpu") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "SentenceTransformerCrossEncoder requires sentence-transformers + torch. "
                "Install: pip install torch sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.device = device
        self._pair_cache = PairScoreCache()
        try:
            self._model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise RerankerError(
                f"SentenceTransformerCrossEncoder init failed for {model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        documents = [_build_doc_text(h) for h in hits]
        # Pair scores are deterministic per (query, doc): only run the model
        # on documents not already scored under this query (the pipeline's
        # decompose/iterative merge passes re-submit a superset).
        found, missing = self._pair_cache.get_many(query.text, documents)
        if missing:
            pairs = [[query.text, documents[i]] for i in missing]
            # Inference is blocking C++; run in a thread to keep the event loop free.
            new_scores = await asyncio.to_thread(self._score_sync, pairs)
            self._pair_cache.put_many(
                query.text, [documents[i] for i in missing], new_scores
            )
            for i, s in zip(missing, new_scores):
                found[i] = float(s)
        raw_scores = [found[i] for i in range(len(documents))]

        ranked: list[RankedHit] = []
        for hit, raw in zip(hits, raw_scores):
            # bge-reranker outputs are typically already in [0, 1] for base
            # via sigmoid-style activation. Clamp defensively.
            score = float(raw)
            # If the model emits raw logits in larger range, apply sigmoid.
            if score < -10 or score > 10:
                score = 1.0 / (1.0 + math.exp(-score))
            score = max(0.0, min(1.0, score))
            ranked.append(RankedHit(
                hit=hit,
                semantic_score=score,
                recency_score=0.0,    # downstream ScoreReranker fills in
                authority_score=hit.record.authority_score,
                pin_boost=1.0 if hit.record.pinned else 0.0,
                final_score=score,
                score_reason=f"st-ce={score:.3f} ({self.model_name})",
            ))
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked[:k]

    def _score_sync(self, pairs: list[list[str]]) -> list[float]:
        try:
            scores = self._model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as exc:
            raise RerankerError(
                f"SentenceTransformerCrossEncoder.predict failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
