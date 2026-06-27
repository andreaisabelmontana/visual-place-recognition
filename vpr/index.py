"""Retrieval index: exact nearest-neighbour search over place fingerprints.

Each database image is stored as a descriptor vector with an associated place
label. A query descriptor is compared against every database vector and the
closest matches are returned.

Honest note: this is *exact* nearest-neighbour search -- every query is scored
against the entire database (O(N) per query). For the small controlled
benchmark here that is instant and gives ground-truth-correct rankings. A real
city-scale system with millions of places would replace this with an
approximate index such as FAISS (IVF/HNSW/PQ) to keep queries fast; the
descriptor and evaluation code would not change.

Two metrics are supported:
  * ``cosine``    -- 1 - cosine similarity (descriptors are L2-normalised, so
    this is monotone with Euclidean distance but kept explicit for clarity).
  * ``euclidean`` -- L2 distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class PlaceIndex:
    """Exact k-NN retrieval index over descriptor vectors."""

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric
        self._nn: NearestNeighbors | None = None
        self._vectors: np.ndarray | None = None
        self.labels: list = []
        self.ids: list = []

    def build(self, vectors: np.ndarray, labels: list, ids: list | None = None) -> "PlaceIndex":
        """Index a stack of descriptor vectors.

        Parameters
        ----------
        vectors : (N, D) float array of descriptors.
        labels  : length-N list of place labels (used to judge correctness).
        ids     : optional length-N list of item identifiers (e.g. filenames).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D (N, D) array")
        if len(labels) != vectors.shape[0]:
            raise ValueError("number of labels must match number of vectors")
        self._vectors = vectors
        self.labels = list(labels)
        self.ids = list(ids) if ids is not None else list(range(vectors.shape[0]))
        self._nn = NearestNeighbors(
            n_neighbors=min(len(labels), max(1, len(labels))), metric=self.metric
        )
        self._nn.fit(vectors)
        return self

    def __len__(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    def query(self, vector: np.ndarray, k: int = 5):
        """Return the top-``k`` nearest database items to ``vector``.

        Returns a list of ``(rank, item_id, label, distance)`` tuples ordered by
        ascending distance (closest first).
        """
        if self._nn is None:
            raise RuntimeError("index not built -- call build() first")
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self))
        dist, idx = self._nn.kneighbors(vector, n_neighbors=k)
        results = []
        for rank, (d, i) in enumerate(zip(dist[0], idx[0])):
            results.append((rank, self.ids[i], self.labels[i], float(d)))
        return results

    def query_labels(self, vector: np.ndarray, k: int = 5) -> list:
        """Convenience: just the ordered list of retrieved place labels."""
        return [label for _, _, label, _ in self.query(vector, k)]
