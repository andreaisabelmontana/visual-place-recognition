"""Glue: turn benchmark items into descriptors and a built retrieval index.

Keeps the demo, evaluation and tests using one consistent description path.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .dataset import Benchmark, Item
from .descriptors import GlobalDescriptor
from .index import PlaceIndex

DescribeFn = Callable[[np.ndarray], np.ndarray]


def _to_descriptor_input(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def describe_items(
    items: list[Item],
    describe: DescribeFn | None = None,
    size: int = 256,
) -> tuple[np.ndarray, list, list]:
    """Describe a list of items.

    Returns ``(vectors, labels, names)`` where ``vectors`` is an (N, D) array.
    """
    describe = describe or GlobalDescriptor(image_size=size)
    vectors, labels, names = [], [], []
    for it in items:
        img = _to_descriptor_input(it.image, size)
        vectors.append(describe(img))
        labels.append(it.label)
        names.append(it.name)
    return np.vstack(vectors).astype(np.float32), labels, names


def build_index(
    bench: Benchmark,
    describe: DescribeFn | None = None,
    metric: str = "cosine",
    size: int = 256,
):
    """Describe the database, build a PlaceIndex, and describe the queries.

    Returns ``(index, query_vectors, query_labels, db_labels)``.
    """
    db_vecs, db_labels, db_names = describe_items(bench.database, describe, size)
    q_vecs, q_labels, _ = describe_items(bench.queries, describe, size)
    index = PlaceIndex(metric=metric).build(db_vecs, db_labels, db_names)
    return index, q_vecs, q_labels, db_labels
