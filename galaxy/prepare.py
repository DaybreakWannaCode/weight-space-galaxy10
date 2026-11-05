"""Build the uint8 image cache, the metadata file and the fixed split from the downloaded HDF5 (notebook steps 2 and 4).

    python galaxy/prepare.py
"""
import os

import numpy as np

from common import CLASS_NAMES, FRACS, H5_PATH, IMG_NPY, META_NPZ, SPLIT_SEED, SPLITS_NPZ, DATA_DIR, make_splits


def build_cache(chunk=256):
    """Copy the images out of the HDF5 file into a .npy array, plus labels and metadata into meta.npz. (Notebook code.)"""
    import h5py
    tmp = DATA_DIR / (IMG_NPY.name + ".tmp")
    with h5py.File(H5_PATH, "r") as f:
        print("h5 keys:", {k: tuple(f[k].shape) for k in f.keys()})
        n = f["images"].shape[0]
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=tuple(f["images"].shape))
        for s in range(0, n, chunk):
            out[s:s + chunk] = f["images"][s:s + chunk]
            print(f"\r  caching {min(s + chunk, n)}/{n}", end="")
        out.flush()
        del out
        print()
        np.savez(
            META_NPZ,
            labels=np.asarray(f["ans"][:]).astype(np.int64),
            ra=np.asarray(f["ra"][:]).astype(np.float64),
            dec=np.asarray(f["dec"][:]).astype(np.float64),
            redshift=np.asarray(f["redshift"][:]).astype(np.float32),
            pxscale=np.asarray(f["pxscale"][:]).astype(np.float32),
        )
    os.replace(tmp, IMG_NPY)


if __name__ == "__main__":
    if not (IMG_NPY.exists() and META_NPZ.exists()):
        build_cache()
    labels = np.load(META_NPZ)["labels"]
    if SPLITS_NPZ.exists():
        splits = dict(np.load(SPLITS_NPZ))
    else:
        splits = make_splits(labels, SPLIT_SEED, FRACS)
        np.savez(SPLITS_NPZ, **splits)
    print({k: len(v) for k, v in splits.items()})
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {c} {name:<26s}{int((labels == c).sum()):>6d}")
