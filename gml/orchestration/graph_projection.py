"""Cluster + graph projection over memory embedding vectors.

The orchestration layer stores memories with dense vectors but has no notion
of *clusters* or a *node/edge graph* — those are presentation concerns the
``/api`` surface needs for the 3D memory graph. This module derives them,
read-only, from the vectors the retriever already holds:

  * **clusters** — KMeans over the vectors (sklearn if present, tiny numpy
    fallback otherwise), labelled by the dominant entity in each cluster.
  * **2D coords** — classical PCA (SVD), used for cluster centroids and as a
    stable layout seed.
  * **edges** — k-nearest-neighbour cosine links above a similarity floor,
    deduplicated to undirected pairs.

Everything is cached and keyed by a *signature* of the current memory set
``(count, ids)`` so it recomputes only when memories are added or removed.
No backend logic is touched; this is a derived view.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orchestration.retriever.semantic_retriever import SemanticRetriever
    from orchestration.pipeline.contracts import MemoryItem

# Maps cluster index → a token from the design system (--cluster-1..6).
CLUSTER_COLORS = [
    "cluster-1",
    "cluster-2",
    "cluster-3",
    "cluster-4",
    "cluster-5",
    "cluster-6",
]

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you",
    "are", "was", "were", "has", "have", "had", "will", "would", "their",
    "they", "them", "uses", "using", "use", "runs", "into", "over",
}


@dataclass
class Projection:
    """Immutable, cached projection of the current memory set."""

    signature: tuple
    cluster_by_id: dict[str, int]
    coords_by_id: dict[str, tuple[float, float]]
    clusters: list[dict]
    edges: list[dict]
    explained_variance_ratio: list[float]


class GraphProjector:
    """Computes and caches a :class:`Projection` for a retriever's memories."""

    def __init__(
        self,
        retriever: "SemanticRetriever",
        *,
        max_clusters: int = 6,
        knn: int = 4,
        edge_threshold: float = 0.45,
    ) -> None:
        self._retriever = retriever
        self._max_clusters = max_clusters
        self._knn = knn
        self._edge_threshold = edge_threshold
        self._cache: Projection | None = None

    @classmethod
    def for_items(cls, records, vectors, **kwargs) -> "GraphProjector":
        """Build a projector from an explicit ``(records, {id: vector})`` set.

        The in-memory (JSONL) path reads vectors off the SemanticRetriever, but
        the Postgres backend keeps them in the DB. This adapts a fetched set to
        the ``.records`` / ``._vectors`` shape ``_compute`` expects, so the same
        PCA/KMeans/kNN math serves both backends.
        """
        from types import SimpleNamespace

        source = SimpleNamespace(records=list(records), _vectors=dict(vectors))
        return cls(source, **kwargs)

    def _signature(self) -> tuple:
        recs = self._retriever.records
        return (len(recs), tuple(r.id for r in recs))

    def get(self) -> Projection:
        """Return the projection, recomputing only if the memory set changed."""
        sig = self._signature()
        if self._cache is None or self._cache.signature != sig:
            self._cache = self._compute(sig)
        return self._cache

    # -- internals ---------------------------------------------------------

    def _compute(self, sig: tuple) -> Projection:
        recs = self._retriever.records
        vecs = self._retriever._vectors
        ids = [r.id for r in recs if r.id in vecs]
        if not ids:
            return Projection(sig, {}, {}, [], [], [0.0, 0.0])

        X = np.array([vecs[i] for i in ids], dtype=np.float64)
        n = len(ids)

        coords_by_id, evr = self._pca_2d(ids, X)
        k = max(1, min(self._max_clusters, n))
        labels = self._kmeans_labels(X, k)
        cluster_by_id = {ids[i]: int(labels[i]) for i in range(n)}

        rec_by_id = {r.id: r for r in recs}
        clusters: list[dict] = []
        for c in range(k):
            members = [ids[i] for i in range(n) if int(labels[i]) == c]
            if not members:
                continue
            cx = float(np.mean([coords_by_id[m][0] for m in members]))
            cy = float(np.mean([coords_by_id[m][1] for m in members]))
            clusters.append(
                {
                    "id": c,
                    "label": self._cluster_label([rec_by_id[m] for m in members]),
                    "centroid": {"x": cx, "y": cy},
                    "size": len(members),
                    "color_hint": CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                }
            )

        edges = self._knn_edges(ids, X)
        return Projection(sig, cluster_by_id, coords_by_id, clusters, edges, evr)

    @staticmethod
    def _pca_2d(
        ids: list[str], X: np.ndarray
    ) -> tuple[dict[str, tuple[float, float]], list[float]]:
        n = len(ids)
        if n < 2:
            return {ids[0]: (0.0, 0.0)} if ids else {}, [0.0, 0.0]
        mean = X.mean(axis=0)
        Xc = X - mean
        _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        if Vt.shape[0] >= 2:
            comp = Vt[:2]
        else:
            comp = np.vstack([Vt, np.zeros((2 - Vt.shape[0], Vt.shape[1]))])
        proj = Xc @ comp.T
        total_var = float((S ** 2).sum()) or 1.0
        evr = [
            float((S[0] ** 2) / total_var) if len(S) else 0.0,
            float((S[1] ** 2) / total_var) if len(S) > 1 else 0.0,
        ]
        coords = {ids[i]: (float(proj[i, 0]), float(proj[i, 1])) for i in range(n)}
        return coords, evr

    def _kmeans_labels(self, X: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return np.zeros(len(X), dtype=int)
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            return km.fit_predict(X)
        except Exception:
            return self._kmeans_numpy(X, k)

    @staticmethod
    def _kmeans_numpy(X: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
        """Deterministic Lloyd's algorithm fallback when sklearn is absent."""
        rng = np.random.default_rng(0)
        n = len(X)
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                pts = X[labels == c]
                if len(pts):
                    centers[c] = pts.mean(axis=0)
        return labels

    def _knn_edges(self, ids: list[str], X: np.ndarray) -> list[dict]:
        n = len(ids)
        kk = min(self._knn, n - 1)
        if kk <= 0:
            return []
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        sims = Xn @ Xn.T
        np.fill_diagonal(sims, -1.0)
        edges: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for i in range(n):
            nbrs = np.argpartition(sims[i], -kk)[-kk:]
            for j in nbrs:
                j = int(j)
                w = float(sims[i, j])
                if w < self._edge_threshold:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                edges.append({"source": ids[a], "target": ids[b], "weight": round(w, 4)})
        return edges

    @staticmethod
    def _cluster_label(members: "list[MemoryItem]") -> str:
        ents = Counter(m.entity for m in members if m.entity)
        if ents:
            return str(ents.most_common(1)[0][0])
        words: Counter = Counter()
        for m in members:
            text = (m.value or m.content or "")
            for raw in text.lower().split():
                w = raw.strip(".,:;!?()[]'\"")
                if len(w) >= 4 and w not in _STOPWORDS:
                    words[w] += 1
        if words:
            return words.most_common(1)[0][0]
        return f"cluster {members[0].id[:6]}" if members else "cluster"
