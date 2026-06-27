"""Materialise a few synthetic base 'place' scenes to data/base/.

These committed base images make the benchmark reproducible without bundling
copyrighted photos. The full DATABASE/QUERY split is derived from them at
runtime by vpr.dataset (mild photometric + geometric augmentation).

Run:  python generate_base_images.py
"""

from __future__ import annotations

import os

import cv2

from vpr.dataset import synth_base_image

OUT_DIR = os.path.join("data", "base")
N_PLACES = 12


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(N_PLACES):
        img = synth_base_image(seed=1000 + i)
        path = os.path.join(OUT_DIR, f"place{i:02d}.png")
        cv2.imwrite(path, img)
        print(f"wrote {path}")
    print(f"\n{N_PLACES} base place images written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
