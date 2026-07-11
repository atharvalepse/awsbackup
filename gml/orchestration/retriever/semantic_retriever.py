"""Real semantic Retriever — uses an Embedder for both queries and records.

Where :class:`StubRetriever` derives record vectors from a SHA-256 hash so
the stub Embedder's queries align with stub record vectors, this retriever
runs records through the same Embedder the Pipeline uses for queries, so
similarity is genuine semantic similarity (not hash-coincidence).

Storage is in-memory: a list of (MemoryItem, vector) pairs plus a tiny
cosine-search loop. For larger corpora swap in a real vector DB behind the
same :class:`Retriever` interface.
"""
import math
import os

from orchestration.embedder.base import Embedder
from orchestration.errors import RetrieverError
from orchestration.pipeline.contracts import (
    Classification,
    ClassificationSource,
    EmbeddedQuery,
    MemoryItem,
    Query,
    RetrievalHit,
    TargetDescriptor,
)
from orchestration.retriever.base import Retriever


# Adversarial-gating threshold (override via env GML_MATCH_THRESHOLD).
#
# Why 0.30, not 0.0:
# LOCOMO has ~22% adversarial questions ("what's our SLA?" when no SLA was
# ever discussed). With threshold 0.0, search() ALWAYS returns memories,
# even garbage matches at cosine ~0.10. The bench's cat-5 gold expects
# "I don't know" / empty context, so any noise leakage costs recall.
#
# A threshold of 0.30 keeps real matches (typically 0.5-0.9 cosine on
# bge-large) while gating away noise. Set 0.0 to disable.
#
# Lowered to 0.20: the retrieval stage stays permissive and the cross-encoder
# reranker (now always run on recall) handles quality filtering. NOTE: this
# also widens the pipeline YES/NO adversarial gate — re-check cat-5 ("I don't
# know") recall on LOCOMO after this change.
DEFAULT_MATCH_THRESHOLD = float(os.environ.get("GML_MATCH_THRESHOLD", "0.20"))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise RetrieverError(
            f"vector dimension mismatch: query={len(a)} record={len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    # Clamp to [-1, 1] — float drift can produce 1.0000000000000002 etc.
    if sim > 1.0:
        return 1.0
    if sim < -1.0:
        return -1.0
    return sim


class SemanticRetriever(Retriever):
    """In-memory cosine retriever backed by a real :class:`Embedder`.

    Records must be ingested via :meth:`ingest` before they can be retrieved
    — this is async because real Embedders are async. After ingestion, the
    retriever has the same shape and interface as :class:`StubRetriever`.

    The ingestion embedder and the query-side Embedder do not have to be
    the same instance, but their ``version`` must match — otherwise queries
    and records live in different vector spaces and similarities are
    meaningless.

    Example:
        >>> retriever = SemanticRetriever(embedder=gemini_embedder)
        >>> await retriever.ingest(default_records())
        >>> # ...pipeline runs as usual
    """

    def __init__(
        self,
        embedder: Embedder,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.records: list[MemoryItem] = []
        self._vectors: dict[str, list[float]] = {}

    async def ingest(self, records: list[MemoryItem]) -> None:
        """Compute and store a vector for each record. Each record's
        ``content`` (plus entity/attribute when present) is embedded with a
        synthetic Query carrying a no-op Classification so the Embedder
        interface stays uniform between ingestion and retrieval."""
        for record in records:
            signal = record.content
            if record.entity:
                signal += " || " + record.entity
                if record.attribute:
                    signal += ":" + record.attribute
            synthetic_query = _synthetic_query(signal)
            embedded = await self.embedder.embed(synthetic_query, _NO_OP_CLASSIFICATION)
            self.records.append(record)
            self._vectors[record.id] = embedded.vector

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        return self._rank(embedded.vector, k=len(self.records), threshold=self.match_threshold)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        return self._rank(embedded.vector, k=k, threshold=self.match_threshold)

    async def get_neighbors(
        self, embedded: EmbeddedQuery, record_id: str, k: int = 3
    ) -> list[RetrievalHit]:
        """1-hop graph expansion: k nearest records to the SEED record's own
        vector (excluding the seed). Same contract as the pgvector variant."""
        seed_vec = self._vectors.get(record_id)
        if seed_vec is None:
            return []
        neighbor_sim = float(os.environ.get("GML_GRAPH_NEIGHBOR_SIM", "0.45"))
        hits = self._rank(seed_vec, k=k + 1, threshold=neighbor_sim)
        return [h for h in hits if h.record.id != record_id][:k]

    def remove(self, memory_ids: "set[str] | list[str]") -> int:
        """Drop records and their vectors by id. Returns the count removed.

        Mirrors a delete in the backing store so the in-memory index stays
        consistent without a full re-embed of the corpus.
        """
        ids = set(memory_ids)
        before = len(self.records)
        self.records = [r for r in self.records if r.id not in ids]
        for mid in ids:
            self._vectors.pop(mid, None)
        return before - len(self.records)

    def _rank(self, query_vec: list[float], k: int, threshold: float) -> list[RetrievalHit]:
        if not self._vectors:
            return []
        scored: list[RetrievalHit] = []
        for record in self.records:
            # Skip superseded memories so they don't surface on recall.
            if not getattr(record, "is_latest", True):
                continue
            sim = _cosine(query_vec, self._vectors[record.id])
            if sim > threshold:
                scored.append(RetrievalHit(record=record, similarity=sim))
        scored.sort(key=lambda h: h.similarity, reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------
# Internal: synthetic Query for record ingestion
# --------------------------------------------------------------------------


_NO_OP_CLASSIFICATION = Classification(
    intent_type="ingest",
    entities=[],
    retrieval_hints={},
    confidence=1.0,
    source=ClassificationSource.FAST_PATH,
)


def _synthetic_target() -> TargetDescriptor:
    # Cheap valid TargetDescriptor — only used to satisfy the Query schema
    # during record ingestion; never reaches an actual target adapter.
    return TargetDescriptor.for_chatgpt()


def _synthetic_query(text: str) -> Query:
    return Query(
        text=text,
        target=_synthetic_target(),
        session_context={},
        trace_id="ingest",
    )
