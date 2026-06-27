"""Visual place recognition demo.

Builds the controlled benchmark, indexes the database with the colour+HOG
global descriptor, runs every query, prints a few example retrievals, and
reports Recall@K against the random-retrieval baseline. Also evaluates the
Bag-of-Visual-Words descriptor for comparison.

Run:  python demo.py
"""

from __future__ import annotations

from vpr.dataset import build_benchmark
from vpr.descriptors import BoVW, GlobalDescriptor
from vpr.evaluate import evaluate
from vpr.pipeline import build_index, describe_items
from vpr.index import PlaceIndex


KS = [1, 3, 5]


def show_example_queries(bench, index, q_vecs, q_labels, n=5):
    print("\nExample retrievals (query -> top-3 retrieved places):")
    print("-" * 60)
    for i in range(min(n, len(bench.queries))):
        q = bench.queries[i]
        results = index.query(q_vecs[i], k=3)
        top = ", ".join(f"place{lab:02d}(d={d:.3f})" for _, _, lab, d in results)
        correct = "OK " if results[0][2] == q_labels[i] else "MISS"
        print(f"[{correct}] {q.name:14s} -> {top}")


def run_descriptor(name, bench, describe, metric="cosine"):
    index, q_vecs, q_labels, db_labels = build_index(bench, describe, metric=metric)
    res = evaluate(index, q_vecs, q_labels, db_labels, ks=KS)
    print(f"\n=== Descriptor: {name} ===")
    print(f"database items: {len(index)} | queries: {res.n_queries} | "
          f"places: {bench.n_places} | dim: {q_vecs.shape[1]}")
    print(res.summary())
    return index, q_vecs, q_labels, res


def main() -> None:
    bench = build_benchmark(n_places=12, db_per_place=4, queries_per_place=3, seed=0)
    print("Controlled VPR benchmark")
    print(f"  places={bench.n_places}  database={bench.n_database}  queries={bench.n_queries}")

    # 1) colour + HOG global descriptor
    index, q_vecs, q_labels, res = run_descriptor(
        "color+HOG (global)", bench, GlobalDescriptor()
    )
    show_example_queries(bench, index, q_vecs, q_labels)

    # 2) Bag of Visual Words over ORB
    bovw = BoVW(vocab_size=64)
    db_imgs = [__resize(it.image) for it in bench.database]
    bovw.fit(db_imgs)
    run_descriptor("BoVW-64 (ORB)", bench, bovw)

    margin = res.recall_at_k[1] - res.random_at_k[1]
    print("\n" + "=" * 60)
    print(f"Headline (color+HOG): Recall@1 = {res.recall_at_k[1]:.3f} vs "
          f"random {res.random_at_k[1]:.3f}  (+{margin:.3f})")
    print("Exact nearest-neighbour retrieval. FAISS would scale this to millions "
          "of places; here it is exact NN over a small controlled benchmark.")


def __resize(image, size=256):
    import cv2
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
