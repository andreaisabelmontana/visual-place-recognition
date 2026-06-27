"""Global image descriptors for visual place recognition.

A descriptor turns an image into a single fixed-length vector -- the "place
fingerprint". Two images of the same place should map to nearby vectors;
images of different places should map to distant ones. Retrieval then becomes
nearest-neighbour search over these vectors (see ``vpr.index``).

Three descriptors are implemented, all classical (no deep nets):

* ``color_histogram`` -- HSV colour distribution, robust-ish to small geometry
  changes but sensitive to lighting.
* ``hog_descriptor``   -- Histogram of Oriented Gradients on a fixed grid,
  captures coarse structure/layout and ignores absolute colour.
* ``BoVW``             -- Bag of Visual Words over ORB keypoints. A KMeans
  vocabulary is learned once on a set of images; each image is then encoded as
  a normalised histogram of visual-word occurrences.

``GlobalDescriptor`` concatenates colour + HOG into one L2-normalised vector,
which is the default fingerprint used by the demo and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans


# --------------------------------------------------------------------------- #
# Image loading helpers
# --------------------------------------------------------------------------- #
def load_image(path: str, size: int = 256) -> np.ndarray:
    """Load an image as BGR uint8 and resize to ``size`` x ``size``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + eps)


# --------------------------------------------------------------------------- #
# Colour histogram descriptor
# --------------------------------------------------------------------------- #
def color_histogram(image: np.ndarray, bins: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """HSV colour histogram, flattened and L2-normalised.

    Length = bins[0] * bins[1] * bins[2] (default 512).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
    )
    return _l2_normalize(hist.flatten())


# --------------------------------------------------------------------------- #
# HOG descriptor (coarse spatial structure)
# --------------------------------------------------------------------------- #
def hog_descriptor(image: np.ndarray, cells: int = 8, orientations: int = 9) -> np.ndarray:
    """Histogram of Oriented Gradients on a ``cells`` x ``cells`` grid.

    Each grid cell contributes an ``orientations``-bin histogram of gradient
    directions weighted by magnitude. Length = cells*cells*orientations.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi)  # unsigned orientation in [0, pi)

    h, w = gray.shape
    ch, cw = h // cells, w // cells
    bin_width = np.pi / orientations
    feats = np.zeros((cells, cells, orientations), dtype=np.float32)

    for r in range(cells):
        for c in range(cells):
            r0, c0 = r * ch, c * cw
            cell_mag = mag[r0:r0 + ch, c0:c0 + cw]
            cell_ang = ang[r0:r0 + ch, c0:c0 + cw]
            idx = np.minimum((cell_ang / bin_width).astype(np.int32), orientations - 1)
            # weighted histogram of orientations within this cell
            feats[r, c] = np.bincount(
                idx.ravel(), weights=cell_mag.ravel(), minlength=orientations
            )[:orientations]

    return _l2_normalize(feats.flatten())


# --------------------------------------------------------------------------- #
# Combined global descriptor (colour + HOG)
# --------------------------------------------------------------------------- #
def global_descriptor(image: np.ndarray) -> np.ndarray:
    """Concatenate colour histogram + HOG into one L2-normalised fingerprint."""
    color = color_histogram(image)
    hog = hog_descriptor(image)
    return _l2_normalize(np.concatenate([color, hog]))


@dataclass
class GlobalDescriptor:
    """Callable wrapper around :func:`global_descriptor` with a stable name."""

    image_size: int = 256

    @property
    def name(self) -> str:
        return "color+hog"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return global_descriptor(image)

    def from_path(self, path: str) -> np.ndarray:
        return self(load_image(path, self.image_size))


# --------------------------------------------------------------------------- #
# Bag of Visual Words over ORB features
# --------------------------------------------------------------------------- #
class BoVW:
    """Bag of Visual Words descriptor built on ORB keypoints.

    Workflow:
      1. ``fit(images)``  -- detect ORB descriptors across all images and run
         KMeans to learn a vocabulary of ``vocab_size`` visual words.
      2. ``transform(image)`` -- assign each ORB descriptor of an image to its
         nearest visual word and build an L2-normalised word-occurrence
         histogram of length ``vocab_size``.
    """

    def __init__(self, vocab_size: int = 64, n_features: int = 400, random_state: int = 0):
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.kmeans: KMeans | None = None

    @property
    def name(self) -> str:
        return f"bovw{self.vocab_size}"

    def _orb_descriptors(self, image: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)
        if desc is None:
            return None
        # ORB descriptors are binary (uint8); KMeans needs floats.
        return desc.astype(np.float32)

    def fit(self, images: list[np.ndarray]) -> "BoVW":
        all_desc = []
        for img in images:
            desc = self._orb_descriptors(img)
            if desc is not None:
                all_desc.append(desc)
        if not all_desc:
            raise ValueError("no ORB descriptors found across the training images")
        stacked = np.vstack(all_desc)
        k = min(self.vocab_size, len(stacked))
        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        self.kmeans.fit(stacked)
        self.vocab_size = k
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("BoVW vocabulary not built -- call fit() first")
        desc = self._orb_descriptors(image)
        hist = np.zeros(self.vocab_size, dtype=np.float32)
        if desc is not None:
            words = self.kmeans.predict(desc)
            for w in words:
                hist[w] += 1.0
        return _l2_normalize(hist)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image)
