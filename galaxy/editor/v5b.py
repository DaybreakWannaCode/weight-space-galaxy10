"""Programmed network v5b, written by the editor (Claude): v5 with the bar-scale measurements fixed.

In v5 the bar-scale ring kernels went through the first ReLU at every pixel, then were averaged over the 4x4 central
pixels: off centre an m=2 kernel mostly sees the galaxy's brightness gradient, so the bar signal was diluted (barred
recall 71% in v4 -> 62% in v5). v5b gives those channels a +25 bias so the first ReLU never clips them, reads the
centre linearly and subtracts 25 there, so the ReLU acts on the centred projection. It also restores v4's ring radii
(2-3 and 3-4.5 px at 64 px) and adds a full-resolution inner elongation (Gaussian sigma 2 px window).

    python galaxy/editor/v5b.py build | knots <acts layer 7> | head <acts layer 14> [--quad] [--temp] [--shrink] --out
"""
import argparse

import numpy as np

import v5
from v5 import CENTRE, KR, KTH, KX, KY, NDIR, PHIS

OFFSET = 25.0


def filters():
    f = v5.filters()
    rA, rB = v5.ring(KR, 2.0, 3.0), v5.ring(KR, 3.0, 4.5)
    win = np.exp(-(KX ** 2 + KY ** 2) / (2 * 2.0 ** 2))
    for j, p in enumerate(PHIS):
        a_, b_ = rA * np.cos(2 * KTH - p), rB * np.cos(2 * KTH - p)
        m_ = win * ((KX ** 2 - KY ** 2) * np.cos(p) + 2 * KX * KY * np.sin(p))
        for name, k in [(f"ringA_{j}", a_), (f"ringB_{j}", b_), (f"turnAB+_{j}", a_ + b_), (f"turnAB-_{j}", a_ - b_),
                        (f"momA_{j}", m_)]:
            f[name] = (np.repeat(k[None] / 3, 3, 0), OFFSET, OFFSET)
    return f


def layout():
    rows, feats = v5.layout()
    rows.extend(("momA_%d" % j, CENTRE) for j in range(NDIR))
    feats.append(("elong_fine_s2", {len(rows) - NDIR + j: 1 / NDIR for j in range(NDIR)}))
    return rows, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "head"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--quad", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v5b_s3")
    a = ap.parse_args()
    if a.stage == "build":
        v5.build("v5b", filters, layout)
    elif a.stage == "knots":
        v5.knots(a.acts, "v5b")
    else:
        v5.head(a.acts, a.quad, a.temp, a.shrink, a.out, "v5b")
