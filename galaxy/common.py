"""Paths, class names, the fixed split and data access for the Galaxy10 weight-programming experiment.

The split logic is copied verbatim from challenge_template.ipynb, so the train/val/test indices match the organizer's.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("GALAXY_DATA_DIR", ROOT.parent / "galaxy10_data"))
H5_PATH = DATA_DIR / "Galaxy10_DECals.h5"
IMG_NPY = DATA_DIR / "images_u8.npy"          # (N, 256, 256, 3) uint8 cache, memory-mapped
META_NPZ = DATA_DIR / "meta.npz"              # labels + per-galaxy metadata
SPLITS_NPZ = DATA_DIR / "splits.npz"          # train / val / test indices

SPLIT_SEED = 1234
FRACS = dict(train=0.80, val=0.10, test=0.10)
NUM_CLASSES = 10
MAX_PARAMS = 500_000                          # the challenge limit: numbers in model.state_dict() (confirmed 500k,
                                              # not the 30M in the notebook's rules cell)

CLASS_NAMES = [
    "Disturbed", "Merging", "Round smooth", "In-between round smooth", "Cigar-shaped smooth",
    "Barred spiral", "Unbarred tight spiral", "Unbarred loose spiral", "Edge-on without bulge", "Edge-on with bulge",
]


def make_splits(labels, seed, fracs):
    """Random split, stratified by class so every split has the same class mix. (Verbatim from the notebook.)"""
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in fracs}
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_train = round(fracs["train"] * len(idx))
        n_val = round(fracs["val"] * len(idx))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}


def load():
    """(images memmap (N, 256, 256, 3) uint8, labels (N,), splits dict of index arrays)."""
    labels = np.load(META_NPZ)["labels"]
    images = np.load(IMG_NPY, mmap_mode="r")
    splits = dict(np.load(SPLITS_NPZ))
    return images, labels, splits


def to_tensor(images, idx, device):
    """Images at sorted indices `idx` as a float32 (B, 3, 256, 256) tensor in [0, 1], as the notebook's Dataset gives."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(device)
    return x.permute(0, 3, 1, 2).float().div_(255.0)
