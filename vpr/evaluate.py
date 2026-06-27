"""Evaluation protocol for visual place recognition.

Given a built :class:`~vpr.index.PlaceIndex` over the database and a set of
query descriptors with known place labels, we measure:

* ``Recall@K`` -- fraction of queries whose true place appears among the top-K
  retrieved database items. Recall@1 is top-1 accuracy.
* A ``random`` baseline -- the expected Recall@K if we returned K database items
  uniformly at random. With ``p`` = (database items of the true place) / (total
  database items), the chance the true place is missed by K independent draws is
  roughly ``(1 - p)**K``, so random Recall@K ~= ``1 - (1 - p)**K``. We also
  estimate it empirically by actually drawing random results, which is the
  number reported.

A useful system must beat the random baseline by a clear margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import PlaceIndex


@dataclass
class EvalResult:
    ks: list[int]
    recall_at_k: dict[int, float]
    random_at_k: dict[int, float]
    n_queries: int

    def top1(self) -> float:
        return self.recall_at_k[1]

    def summary(self) -> str:
        lines = [f"queries: {self.n_queries}"]
        for k in self.ks:
            lines.append(
                f"  Recall@{k:<2d} = {self.recall_at_k[k]:.3f}   "
                f"(random {self.random_at_k[k]:.3f})"
            )
        return "\n".join(lines)


def recall_at_k(
    index: PlaceIndex,
    query_vectors: np.ndarray,
    query_labels: list,
    ks: list[int] | None = None,
) -> dict[int, float]:
    """Compute Recall@K for each K in ``ks`` over the given queries."""
    ks = ks or [1, 5]
    max_k = min(max(ks), len(index))
    hits = {k: 0 for k in ks}
    for vec, true_label in zip(query_vectors, query_labels):
        retrieved = index.query_labels(vec, k=max_k)
        for k in ks:
            if true_label in retrieved[:k]:
                hits[k] += 1
    n = len(query_labels)
    return {k: hits[k] / n for k in ks}


def random_baseline(
    db_labels: list,
    query_labels: list,
    ks: list[int] | None = None,
    trials: int = 200,
    seed: int = 0,
) -> dict[int, float]:
    """Empirical Recall@K of returning K random database items per query."""
    ks = ks or [1, 5]
    rng = np.random.default_rng(seed)
    db_labels = np.asarray(db_labels)
    n_db = len(db_labels)
    max_k = min(max(ks), n_db)
    hits = {k: 0 for k in ks}
    total = 0
    for true_label in query_labels:
        for _ in range(trials):
            picks = db_labels[rng.choice(n_db, size=max_k, replace=False)]
            for k in ks:
                if true_label in picks[:k]:
                    hits[k] += 1
            total += 1
    total_per_k = total  # each (query, trial) counted once
    return {k: hits[k] / total_per_k for k in ks}


def evaluate(
    index: PlaceIndex,
    query_vectors: np.ndarray,
    query_labels: list,
    db_labels: list,
    ks: list[int] | None = None,
    seed: int = 0,
) -> EvalResult:
    """Run the full evaluation: Recall@K plus the random baseline."""
    ks = ks or [1, 5]
    ks = [k for k in ks if k <= len(index)]
    rk = recall_at_k(index, query_vectors, query_labels, ks)
    rnd = random_baseline(db_labels, query_labels, ks, seed=seed)
    return EvalResult(ks=ks, recall_at_k=rk, random_at_k=rnd, n_queries=len(query_labels))
