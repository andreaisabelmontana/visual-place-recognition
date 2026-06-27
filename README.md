# Visual Place Recognition

Tell where a photo was taken by matching it against a database of known places —
the relocalisation trick a self-driving car or a phone uses to find itself when
GPS drops out. Recognition is reframed as **image retrieval**: describe each
image with a compact "place fingerprint", index the database, and answer a query
by retrieving its nearest neighbours.

This is a **classical** implementation (OpenCV + NumPy + scikit-learn). No deep
networks, no FAISS — see the honest note at the bottom.

- Live page: https://andreaisabelmontana.github.io/visual-place-recognition/

## Pipeline

```
image ──► global descriptor ──► exact NN retrieval ──► ranked places ──► Recall@K
          (color hist + HOG,      (cosine / L2 over
           or BoVW over ORB)       the database)
```

### Descriptors — `vpr/descriptors.py`
- **Colour histogram** — 8×8×8 HSV histogram (512-d), L2-normalised.
- **HOG** — Histogram of Oriented Gradients on an 8×8 cell grid, 9 orientations
  (576-d): coarse layout/structure, colour-invariant.
- **Global descriptor** — colour + HOG concatenated and L2-normalised (1088-d).
  This is the default fingerprint.
- **Bag of Visual Words** — ORB keypoints quantised against a KMeans vocabulary
  (default 64 words); each image becomes an L2-normalised word-occurrence
  histogram.

### Retrieval index — `vpr/index.py`
Exact k-nearest-neighbour search (`sklearn.neighbors.NearestNeighbors`) over the
database descriptors, cosine or Euclidean. Every query is scored against the
whole database, so rankings are exact (ground-truth correct) for this size.

### Dataset — `vpr/dataset.py` + `generate_base_images.py`
A **controlled, synthetic benchmark**, not real-world imagery. 12 distinct
textured "place" scenes are generated as base images (committed under
`data/base/`; drop your own photos there and they are used instead). From each
base we derive **DATABASE** and **QUERY** views with mild photometric/geometric
augmentation — brightness/contrast, blur, small rotation, crop, perspective
warp — using disjoint random seeds so a query is never an exact copy of a
database view. A query of place X should retrieve database views of place X.

### Evaluation — `vpr/evaluate.py`
**Recall@K** = fraction of queries whose true place appears in the top-K
retrieved items (Recall@1 = top-1 accuracy). Compared against an empirical
**random-retrieval baseline** (returning K random database items).

## Results

Benchmark: 12 places, 48 database views, 36 queries. Real numbers from
`python demo.py`:

| Descriptor    | dim  | Recall@1 | Recall@3 | Recall@5 |
|---------------|------|----------|----------|----------|
| color+HOG     | 1088 | **0.944**| 1.000    | 1.000    |
| BoVW-64 (ORB) | 64   | **1.000**| 1.000    | 1.000    |
| random        | —    | 0.086    | 0.240    | 0.368    |

color+HOG top-1 beats the random baseline by **+0.858** (0.944 vs 0.086).

## Run it

```bash
pip install -r requirements.txt
python generate_base_images.py   # writes data/base/*.png (already committed)
python demo.py                   # retrieval examples + Recall@K vs random
python -m pytest -q              # 7 tests
```

Tests assert that a same-place augmented view is closer than a different place,
that top-1 retrieval is correct for clear queries (≥0.9), that Recall@K beats the
random baseline by a clear margin, and that the BoVW vocabulary builds and
assigns visual words.

## Honest note: classical vs deep, exact vs approximate

State-of-the-art VPR uses **learned** global descriptors (NetVLAD, CNN/ViT
embeddings) that survive lighting, season and viewpoint changes far better than
the hand-crafted descriptors here, and **approximate** indexes (FAISS — IVF /
HNSW / PQ) to search millions of places fast. This project deliberately uses
neither: it is colour+HOG / BoVW descriptors with **exact** nearest-neighbour
search on a **small controlled benchmark**. That keeps it fully runnable from a
clean `pip install` and makes the retrieve-then-score pipeline transparent. The
strong Recall@K here reflects a controlled benchmark — not the appearance-change
difficulty of a real-world deployment. Swapping in deep descriptors and FAISS
would not change the evaluation code.
