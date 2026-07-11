"""Bounded LRU cache for cross-encoder (query, document) raw pair scores.

Why: the pipeline reranks up to three times per request (initial pass,
query-decompose merge, iterative-retrieval merge). The merge passes score a
superset of the candidates already scored under the SAME query text, so
without caching the same (query, doc) pair hits the cross-encoder 2-3x.
Pair scores are deterministic per (query, doc), so memoizing the raw model
output is semantics-preserving — sigmoid/clamping happens downstream as
before.

The cache also persists across requests (bounded LRU), so repeated queries
over a stable corpus skip the model entirely.

Size via GML_CE_CACHE_SIZE (entries; one float per entry). 0 disables.
"""
import hashlib
import os
from collections import OrderedDict


def _default_max_entries() -> int:
    return int(os.environ.get("GML_CE_CACHE_SIZE", "4096"))


class PairScoreCache:
    """LRU of sha1(query \\x1f doc) -> raw model score. Not thread-safe by
    design: all callers touch it from the event loop; only the model forward
    pass itself runs in a worker thread."""

    def __init__(self, max_entries: int | None = None) -> None:
        self.max_entries = (
            _default_max_entries() if max_entries is None else max_entries
        )
        self._store: OrderedDict[str, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(query: str, doc: str) -> str:
        h = hashlib.sha1()
        h.update(query.encode("utf-8"))
        h.update(b"\x1f")
        h.update(doc.encode("utf-8"))
        return h.hexdigest()

    def get_many(
        self, query: str, docs: list[str]
    ) -> tuple[dict[int, float], list[int]]:
        """Return ({index: cached_score}, [indices_missing])."""
        if self.max_entries <= 0:
            return {}, list(range(len(docs)))
        found: dict[int, float] = {}
        missing: list[int] = []
        for i, doc in enumerate(docs):
            key = self._key(query, doc)
            score = self._store.get(key)
            if score is None:
                missing.append(i)
            else:
                self._store.move_to_end(key)
                found[i] = score
        self.hits += len(found)
        self.misses += len(missing)
        return found, missing

    def put_many(self, query: str, docs: list[str], scores: list[float]) -> None:
        if self.max_entries <= 0:
            return
        for doc, score in zip(docs, scores):
            key = self._key(query, doc)
            self._store[key] = float(score)
            self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
