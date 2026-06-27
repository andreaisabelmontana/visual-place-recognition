"""Visual Place Recognition -- classical descriptor + exact NN retrieval.

A small, honest visual place recognition system: describe each image with a
classical global descriptor (colour histogram + HOG, or a Bag-of-Visual-Words
over ORB), index the database with exact nearest-neighbour search, and answer
"where was this taken?" by retrieving the closest known places.
"""

from .descriptors import (
    BoVW,
    GlobalDescriptor,
    color_histogram,
    global_descriptor,
    hog_descriptor,
    load_image,
)
from .index import PlaceIndex
from .evaluate import EvalResult, evaluate, random_baseline, recall_at_k
from .dataset import Benchmark, Item, build_benchmark, synth_base_image
from .pipeline import build_index, describe_items

__all__ = [
    "BoVW",
    "GlobalDescriptor",
    "color_histogram",
    "global_descriptor",
    "hog_descriptor",
    "load_image",
    "PlaceIndex",
    "EvalResult",
    "evaluate",
    "random_baseline",
    "recall_at_k",
    "Benchmark",
    "Item",
    "build_benchmark",
    "synth_base_image",
    "build_index",
    "describe_items",
]
