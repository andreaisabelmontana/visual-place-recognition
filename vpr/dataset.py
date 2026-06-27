"""Controlled visual-place-recognition benchmark generator.

This is a *synthetic, controlled* benchmark, not a real-world dataset -- it is
designed so the retrieval pipeline can be tested deterministically and honestly.

Each "place" is a distinct textured scene. From a handful of base place images
we generate multiple DATABASE views and multiple QUERY views by applying mild
photometric and geometric augmentations (brightness/contrast shifts, blur,
small rotation, crop, perspective warp, JPEG-like noise). A query view of place
X should therefore retrieve database views of place X -- exactly the
relocalisation task, but with known ground truth.

If real base images are present in ``data/base/`` they are used; otherwise a set
of synthetic, visually distinct base scenes is generated so the repo is
self-contained.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np


BASE_SIZE = 384  # base images are generated/loaded larger than the 256 used for descriptors


# --------------------------------------------------------------------------- #
# Synthetic base scenes -- each is a distinct, recognisable "place"
# --------------------------------------------------------------------------- #
def _scene_palette(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    top = rng.integers(40, 220, size=3)
    bottom = rng.integers(40, 220, size=3)
    return top.astype(np.float32), bottom.astype(np.float32)


def synth_base_image(seed: int, size: int = BASE_SIZE) -> np.ndarray:
    """Generate one distinct synthetic 'place' scene deterministically from seed.

    Combines a colour gradient sky/ground, a horizon line, a skyline of
    rectangular 'buildings', and scattered structural marks. Different seeds give
    visually very different scenes, so descriptors can tell them apart.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)

    # vertical colour gradient (sky -> ground)
    top, bottom = _scene_palette(rng)
    for y in range(size):
        t = y / (size - 1)
        img[y, :, :] = (1 - t) * top + t * bottom

    # horizon
    horizon = int(rng.integers(int(size * 0.45), int(size * 0.65)))
    ground = rng.integers(30, 160, size=3).astype(np.float32)
    img[horizon:, :, :] = 0.6 * img[horizon:, :, :] + 0.4 * ground

    # skyline of buildings sitting on the horizon
    n_buildings = int(rng.integers(5, 11))
    x = 0
    while x < size and n_buildings > 0:
        bw = int(rng.integers(size // 14, size // 6))
        bh = int(rng.integers(size // 8, size // 3))
        color = rng.integers(20, 200, size=3).astype(np.float32)
        x0, x1 = x, min(size, x + bw)
        y0 = max(0, horizon - bh)
        img[y0:horizon, x0:x1, :] = color
        # a few windows
        for _ in range(int(rng.integers(2, 8))):
            wy = int(rng.integers(y0, horizon - 2)) if horizon - 2 > y0 else y0
            wx = int(rng.integers(x0, max(x0 + 1, x1 - 2)))
            img[wy:wy + 3, wx:wx + 3, :] = rng.integers(180, 255, size=3).astype(np.float32)
        x = x1 + int(rng.integers(2, size // 20))
        n_buildings -= 1

    # scattered structural marks / foreground objects
    for _ in range(int(rng.integers(8, 20))):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(horizon, size))
        r = int(rng.integers(3, 14))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(img, (cx, cy), r, color, -1)

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Augmentations (one view = one augmented copy of a base place)
# --------------------------------------------------------------------------- #
def _adjust_brightness_contrast(img, rng):
    alpha = float(rng.uniform(0.75, 1.25))   # contrast
    beta = float(rng.uniform(-30, 30))       # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _blur(img, rng):
    k = int(rng.choice([1, 3, 5]))
    if k <= 1:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)


def _rotate(img, rng, max_deg=10):
    h, w = img.shape[:2]
    angle = float(rng.uniform(-max_deg, max_deg))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def _crop_resize(img, rng, max_crop=0.12):
    h, w = img.shape[:2]
    cx = int(w * rng.uniform(0, max_crop))
    cy = int(h * rng.uniform(0, max_crop))
    cropped = img[cy:h - cy, cx:w - cx]
    if cropped.size == 0:
        return img
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _perspective(img, rng, jitter=0.06):
    h, w = img.shape[:2]
    j = jitter
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, size=src.shape).astype(np.float32) * [w, h]
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply a mild, random photometric+geometric augmentation to ``img``."""
    rng = np.random.default_rng(seed)
    out = img
    out = _adjust_brightness_contrast(out, rng)
    out = _blur(out, rng)
    out = _rotate(out, rng)
    out = _crop_resize(out, rng)
    out = _perspective(out, rng)
    return out


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    label: int      # place id
    role: str       # "database" or "query"
    image: np.ndarray
    name: str


@dataclass
class Benchmark:
    database: list[Item] = field(default_factory=list)
    queries: list[Item] = field(default_factory=list)
    n_places: int = 0

    @property
    def n_database(self) -> int:
        return len(self.database)

    @property
    def n_queries(self) -> int:
        return len(self.queries)


def load_base_images(base_dir: str, n_places: int) -> list[np.ndarray]:
    """Load real base images from ``base_dir`` if any, else synthesize them."""
    bases: list[np.ndarray] = []
    if os.path.isdir(base_dir):
        files = sorted(
            f for f in os.listdir(base_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        for f in files[:n_places]:
            img = cv2.imread(os.path.join(base_dir, f), cv2.IMREAD_COLOR)
            if img is not None:
                bases.append(cv2.resize(img, (BASE_SIZE, BASE_SIZE)))
    # top up with synthetic scenes to reach n_places
    while len(bases) < n_places:
        bases.append(synth_base_image(seed=1000 + len(bases)))
    return bases[:n_places]


def build_benchmark(
    n_places: int = 12,
    db_per_place: int = 4,
    queries_per_place: int = 3,
    base_dir: str = "data/base",
    seed: int = 0,
) -> Benchmark:
    """Build the full DATABASE/QUERY benchmark from base place images.

    For each place we emit ``db_per_place`` database views and
    ``queries_per_place`` query views, all augmentations of the same base scene
    but generated with disjoint random seeds so query != any database view.
    """
    bases = load_base_images(base_dir, n_places)
    bench = Benchmark(n_places=n_places)

    for place, base in enumerate(bases):
        for j in range(db_per_place):
            s = seed * 100000 + place * 1000 + j
            img = augment(base, seed=s)
            bench.database.append(
                Item(label=place, role="database", image=img, name=f"place{place:02d}_db{j}")
            )
        for j in range(queries_per_place):
            # offset query seeds far from database seeds -> different views
            s = seed * 100000 + place * 1000 + 500 + j
            img = augment(base, seed=s)
            bench.queries.append(
                Item(label=place, role="query", image=img, name=f"place{place:02d}_q{j}")
            )

    return bench
