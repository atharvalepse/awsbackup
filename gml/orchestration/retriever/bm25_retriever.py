"""BM25 lexical retriever — fast keyword-driven recall.

Pure-Python BM25 implementation via the ``rank_bm25`` library. Complementary
to dense (vector) retrieval: BM25 catches exact-term matches that dense
embeddings sometimes smooth over (e.g. specific identifiers, version
strings, rare entity names). When fused with dense retrieval (see
:class:`HybridRetriever`), this consistently lifts recall@k on memory
benchmarks like LOCOMO.

Records are indexed at construction / ``ingest`` time. The search side
ignores ``EmbeddedQuery.vector`` entirely — only the original query text
matters.
"""
import re

from rank_bm25 import BM25Okapi

from orchestration.pipeline.contracts import EmbeddedQuery, MemoryItem, RetrievalHit
from orchestration.retriever.base import Retriever


_TOKEN_RE = re.compile(r"\b[\w]+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _record_to_terms(record: MemoryItem) -> list[str]:
    parts = [record.content]
    if record.entity:
        parts.append(record.entity)
    if record.attribute:
        parts.append(record.attribute)
    if record.value:
        parts.append(record.value)
    if record.summary_short:
        parts.append(record.summary_short)
    return _tokenize(" ".join(parts))


class BM25Retriever(Retriever):
    """Lexical retriever using BM25 over MemoryItem content + structured fields.

    Index updates require a full rebuild (BM25Okapi has no incremental API).
    For our scale (thousands of records) this is fine; for bigger corpora
    swap in a real BM25 server.
    """

    def __init__(self, records: list[MemoryItem] | None = None) -> None:
        self.records: list[MemoryItem] = []
        self._bm25: BM25Okapi | None = None
        self._doc_terms: list[list[str]] = []
        if records:
            self.ingest(records)

    def ingest(self, records: list[MemoryItem]) -> None:
        """Append records and rebuild the index. Idempotent for duplicates by id."""
        seen = {r.id for r in self.records}
        for r in records:
            if r.id in seen:
                continue
            self.records.append(r)
            self._doc_terms.append(_record_to_terms(r))
            seen.add(r.id)
        # rank_bm25 doesn't support empty corpus
        if self._doc_terms:
            self._bm25 = BM25Okapi(self._doc_terms)

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        return self._search_text(embedded.query.text, k=len(self.records))

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        return self._search_text(embedded.query.text, k=k)

    def _search_text(self, text: str, k: int) -> list[RetrievalHit]:
        if not self._bm25 or not self.records:
            return []
        query_terms = _tokenize(text)
        if not query_terms:
            return []
        scores = self._bm25.get_scores(query_terms)
        # BM25 scores can be large positives — normalize to [0, 1]-ish via
        # max-score division so they coexist with cosine similarities.
        max_score = max(scores) if any(s > 0 for s in scores) else 1.0
        hits: list[RetrievalHit] = []
        for record, raw_score in zip(self.records, scores):
            if raw_score <= 0:
                continue
            normalized = min(1.0, raw_score / max_score) if max_score > 0 else 0.0
            hits.append(RetrievalHit(record=record, similarity=normalized))
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:k]
