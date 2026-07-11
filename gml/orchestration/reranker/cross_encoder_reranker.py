"""Cross-encoder reranker — uses fastembed's reranker for true semantic
relevance scoring of query-document pairs.

A bi-encoder (StubEmbedder / FastEmbedEmbedder / GeminiEmbedder) embeds
query and documents independently then cosines them — fast, but loses
joint signal. A cross-encoder scores (query, document) pairs jointly with
a transformer, giving much sharper relevance. The tradeoff is latency:
~100ms for 10 docs vs sub-millisecond for cosine.

This reranker is wired as the SECOND pass after Retriever returns ~50
candidates: take those 50, cross-encode, keep top-10 — then feed to the
score-based Reranker that does the final final_score weighting.

Default model: ``Xenova/ms-marco-MiniLM-L-6-v2`` (~90MB), strong MTEB
score for its size. Swap to ``BAAI/bge-reranker-large`` for higher quality.
"""
import asyncio

from orchestration.errors import RerankerError
from orchestration.pipeline.contracts import Query, RankedHit, RetrievalHit
from orchestration.reranker.base import Reranker
from orchestration.reranker.pair_cache import PairScoreCache


import os

# Default model. Upgrade chain (small → large):
#   Xenova/ms-marco-MiniLM-L-6-v2          (80MB,  ~baseline BEIR)
#   Xenova/ms-marco-MiniLM-L-12-v2         (120MB, +3-5% BEIR)
#   jinaai/jina-reranker-v2-base-multilingual (1.1GB, +8-12% BEIR)  ← current default
#   BAAI/bge-reranker-base                 (1GB)  — fastembed has a broken
#       ONNX file for this in current versions; revisit when fixed.
#   BAAI/bge-reranker-large                (NOT in fastembed list; needs
#       sentence-transformers + torch, ~3GB install)
#
# Note: jina-v2 produces sigmoid-normalized scores in [0,1] but with
# DIFFERENT distribution from L-12. Top-1 relevant scores typically
# 0.5-0.7 instead of 0.9+. SAM-skip threshold (in pipeline.py) auto-tuned
# via GML_SAM_SKIP_THRESHOLD env var.
DEFAULT_MODEL = os.environ.get("GML_CE_MODEL", "jinaai/jina-reranker-v2-base-multilingual")


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranker via fastembed's :class:`Rerank` module.

    Returns RankedHit objects where ``semantic_score`` is the cross-encoder's
    relevance score (sigmoid-normalized to [0, 1]) and the other axes
    (recency/authority/pin) are passed through from the upstream
    :class:`RetrievalHit`. ``final_score`` here is just the cross-encoder
    relevance — the score-based :class:`ScoreReranker` runs after for the
    full weighting.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except Exception as exc:
            raise RerankerError(
                "CrossEncoderReranker requires fastembed>=0.3 with cross-encoder support. "
                f"Import failed: {type(exc).__name__}: {exc}"
            ) from exc

        self.model_name = model_name
        self._pair_cache = PairScoreCache()
        try:
            self._model = TextCrossEncoder(model_name=model_name)
        except Exception as exc:
            raise RerankerError(
                f"CrossEncoderReranker init failed for {model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        documents = [_doc_text(h) for h in hits]
        # Pair scores are deterministic per (query, doc): only run the model
        # on documents not already scored under this query (the pipeline's
        # decompose/iterative merge passes re-submit a superset).
        found, missing = self._pair_cache.get_many(query.text, documents)
        if missing:
            missing_docs = [documents[i] for i in missing]
            new_scores = await asyncio.to_thread(
                self._score_sync, query.text, missing_docs
            )
            self._pair_cache.put_many(query.text, missing_docs, new_scores)
            for i, s in zip(missing, new_scores):
                found[i] = float(s)
        scores = [found[i] for i in range(len(documents))]

        ranked: list[RankedHit] = []
        for hit, raw_score in zip(hits, scores):
            # Sigmoid normalize (cross-encoder outputs are unbounded logits)
            import math
            normalized = 1.0 / (1.0 + math.exp(-raw_score))
            ranked.append(RankedHit(
                hit=hit,
                semantic_score=normalized,
                recency_score=0.0,    # downstream ScoreReranker fills these in
                authority_score=hit.record.authority_score,
                pin_boost=1.0 if hit.record.pinned else 0.0,
                final_score=normalized,
                score_reason=f"cross-encoder={normalized:.3f} (raw={raw_score:.2f})",
            ))

        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked[:k]

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        try:
            scores = list(self._model.rerank(query, documents))
            return [float(s) for s in scores]
        except Exception as exc:
            raise RerankerError(
                f"CrossEncoderReranker.rerank failed: {type(exc).__name__}: {exc}"
            ) from exc


def _doc_text(hit: RetrievalHit) -> str:
    """Build the document text passed to the cross-encoder. Includes
    structured entity/attribute when present so the model sees the schema."""
    parts = [hit.record.content]
    if hit.record.entity:
        prefix = hit.record.entity
        if hit.record.attribute:
            prefix += f": {hit.record.attribute}"
        if hit.record.value:
            prefix += f" = {hit.record.value}"
        parts.append(f"[{prefix}]")
    return " ".join(parts)
