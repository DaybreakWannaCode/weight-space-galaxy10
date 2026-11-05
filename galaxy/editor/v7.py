"""Programmed network v7, written by the editor (Claude): v6 plus how the light is shared among pixels.

Gini / M20-style statistics, the standard non-parametric measures for disturbed and merging galaxies, need the
distribution of pixel brightness regardless of position. For 5 thresholds t (3, 6, 12, 24, 45 sky-noise sigmas above
the sky level measured in pass 1):
  - isophotal area: sum over pixels of clamp(k (L - t), 0, 1), built as relu(k(L-t)) - relu(k(L-t) - 1) from two
    filters (k = 1 / sigma), the difference taken in the feature-sum layer;
  - light above the threshold: sum of relu(L - t);
  - spread of that light: sum of relu(L - t) r^2 (M20-like: bright knots far from the centre).
The areas also give each galaxy's size at fixed surface brightness, which the fixed circles do not.

    python galaxy/editor/v7.py build | knots <acts layer 8> | lda <acts layer 14> | qda <acts layer 14> [options]
"""
import argparse

import numpy as np

import v6  # noqa: F401  (imports v5c: the 29x29 centred grid)
import v2
import v5
import v5b

SKY, SIG = 0.1277, 0.0144            # pass 1: median sky level and noise of the 4x-pooled luminance
LEVELS = [3, 6, 12, 24, 45]


def filters():
    f = v5b.filters()
    lum = np.zeros((3, 9, 9))
    lum[:, 4, 4] = 1 / 3
    k = 1 / SIG
    for n in LEVELS:
        t = SKY + n * SIG
        f[f"area{n}_a"] = (lum * k, -k * t)
        f[f"area{n}_b"] = (lum * k, -k * t - 1)
        f[f"above{n}"] = (lum, -t)
    return f


def layout():
    rows, feats = v6.layout()
    D = v5.disk(v5.RMAX)
    for n in LEVELS:
        rows.append((f"area{n}_a", D))
        rows.append((f"area{n}_b", D))
        feats.append((f"area{n}", {len(rows) - 2: 1.0, len(rows) - 1: -1.0}))
        rows.append((f"above{n}", D))
        feats.append((f"flux_above{n}", {len(rows) - 1: 1.0}))
        rows.append((f"above{n}", D * v5.R ** 2))
        feats.append((f"spread_above{n}", {len(rows) - 1: 1.0}))
    return rows, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "lda", "qda"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--nq", type=int, default=9)
    ap.add_argument("--out", default="v7_s3")
    a = ap.parse_args()
    if a.stage == "build":
        v6.build("v7", layout, filters)
    elif a.stage == "knots":
        v6.knots(a.acts, "v7")
    elif a.stage == "lda":
        v6.lda(a.acts, a.temp, a.shrink, a.out, "v7")
    else:
        v6.qda(a.acts, a.temp, a.alpha, a.shrink, a.out, a.nq, 0.0, "v7")
