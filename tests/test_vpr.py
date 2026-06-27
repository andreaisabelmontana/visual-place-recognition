"""Tests for the visual place recognition pipeline.

Covered:
  * a place's descriptor is closer to an augmented view of the SAME place than
    to a different place;
  * top-1 retrieval returns the correct place for clear queries;
  * Recall@K beats the random baseline by a clear margin;
  * the BoVW vocabulary builds and assigns visual words.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpr.dataset import augment, build_benchmark, synth_base_image
from vpr.descriptors import BoVW, GlobalDescriptor, global_descriptor
from vpr.evaluate import evaluate
from vpr.pipeline import build_index, describe_items


def _resize(img, size=256):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="module")
def bench():
    return build_benchmark(n_places=12, db_per_place=4, queries_per_place=3, seed=0)


def _cos_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def test_same_place_descriptor_is_closer():
    """Descriptor of an augmented same-place view < a different place's view."""
    base_a = synth_base_image(seed=1000)
    base_b = synth_base_image(seed=1007)

    d_a = global_descriptor(_resize(base_a))
    d_a_aug = global_descriptor(_resize(augment(base_a, seed=42)))
    d_b = global_descriptor(_resize(base_b))

    same = _cos_dist(d_a, d_a_aug)
    diff = _cos_dist(d_a, d_b)
    assert same < diff, f"same-place dist {same:.3f} should be < diff {diff:.3f}"


def test_same_place_closer_across_all_places():
    """A single augmented query's nearest single base view is its own place.

    This is the harder single-reference case (one base view per place, no
    database redundancy). We require a clear majority rather than perfection,
    which mirrors the headline retrieval result where multiple database views
    per place push top-1 accuracy higher.
    """
    n = 12
    bases = [synth_base_image(seed=1000 + i) for i in range(n)]
    base_desc = [global_descriptor(_resize(b)) for b in bases]

    wins = 0
    for i, b in enumerate(bases):
        q = global_descriptor(_resize(augment(b, seed=500 + i)))
        dists = [_cos_dist(q, d) for d in base_desc]
        if int(np.argmin(dists)) == i:
            wins += 1
    # far above the 1/12 random chance; clear majority of single-view matches
    assert wins >= 9, f"only {wins}/{n} queries matched their own place"


def test_top1_retrieval_correct(bench):
    """Top-1 retrieval returns the correct place for clear queries (strong)."""
    index, q_vecs, q_labels, db_labels = build_index(bench, GlobalDescriptor())
    correct = sum(
        index.query(v, k=1)[0][2] == lab for v, lab in zip(q_vecs, q_labels)
    )
    acc = correct / len(q_labels)
    assert acc >= 0.9, f"top-1 accuracy {acc:.3f} below 0.9"


def test_recall_beats_random(bench):
    """Recall@K must beat the random-retrieval baseline by a clear margin."""
    index, q_vecs, q_labels, db_labels = build_index(bench, GlobalDescriptor())
    res = evaluate(index, q_vecs, q_labels, db_labels, ks=[1, 5])
    assert res.recall_at_k[1] >= res.random_at_k[1] + 0.5
    assert res.recall_at_k[5] >= res.random_at_k[5] + 0.3
    # Recall is monotone non-decreasing in K
    assert res.recall_at_k[5] >= res.recall_at_k[1]


def test_bovw_builds_and_assigns(bench):
    """BoVW vocabulary builds and produces a valid normalised word histogram."""
    db_imgs = [_resize(it.image) for it in bench.database]
    bovw = BoVW(vocab_size=32)
    bovw.fit(db_imgs)
    assert bovw.kmeans is not None
    assert bovw.vocab_size <= 32 and bovw.vocab_size > 0

    hist = bovw.transform(db_imgs[0])
    assert hist.shape == (bovw.vocab_size,)
    assert np.all(hist >= 0)
    # non-empty image should activate at least one visual word
    assert hist.sum() > 0
    # L2-normalised
    assert abs(np.linalg.norm(hist) - 1.0) < 1e-4


def test_bovw_retrieval_beats_random(bench):
    """BoVW descriptor also retrieves better than random."""
    db_imgs = [_resize(it.image) for it in bench.database]
    bovw = BoVW(vocab_size=64)
    bovw.fit(db_imgs)
    index, q_vecs, q_labels, db_labels = build_index(bench, bovw)
    res = evaluate(index, q_vecs, q_labels, db_labels, ks=[1, 5])
    assert res.recall_at_k[5] > res.random_at_k[5]


def test_index_query_shapes(bench):
    """Index returns ranked results of the requested size."""
    index, q_vecs, q_labels, db_labels = build_index(bench, GlobalDescriptor())
    results = index.query(q_vecs[0], k=5)
    assert len(results) == 5
    ranks = [r[0] for r in results]
    assert ranks == [0, 1, 2, 3, 4]
    dists = [r[3] for r in results]
    assert dists == sorted(dists), "results must be ordered by ascending distance"
