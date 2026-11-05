"""Contact sheet of training galaxies, a few per class, for the editor to look at (rule 3: training data only).

    python galaxy/editor/look.py <out.png> [--per 6] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CLASS_NAMES, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--per", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
images, labels, splits = load()
tr = splits["train"]
rng = np.random.default_rng(a.seed)
fig, axes = plt.subplots(10, a.per, figsize=(1.6 * a.per, 16.5))
for c in range(10):
    pick = rng.choice(tr[labels[tr] == c], a.per, replace=False)
    for k, j in enumerate(pick):
        ax = axes[c, k]
        ax.imshow(images[j])
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(f"{c} {CLASS_NAMES[c]}", fontsize=7, rotation=0, ha="right", va="center")
plt.subplots_adjust(left=0.2, wspace=0.03, hspace=0.03)
plt.savefig(a.out, dpi=70)
print("wrote", a.out)
