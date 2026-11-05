"""Programmed network v8, written by the editor (Claude): v7 repacked on a 27x27 map, plus area statistics of the
fine structure and of the blue and red light.

Repack: every measurement lies within r = 13.5 (32-px units), so the 29x29 map's outer ring is never used. A 13x13
first conv (no padding) gives a 52 px map; AvgPool2d(2, padding=1) then gives 27x27 with pixel (13, 13) exactly on
the galaxy centre. Each measurement's weight map shrinks from 841 to 729 numbers (~50k saved, ~20k spent on the larger
first-conv kernels; the old 9x9 kernels are zero-padded, so every filter computes the same thing as in v7).

New (as v7's luminance areas, clamp(k (x - t), 0, 1) from two filters):
  - coarse band-pass HPc: area above 3, 6, 12 noise sigmas: how much of the galaxy carries arms/clumps (coverage),
    as opposed to the band-pass energy, which one bright knot or star can dominate;
  - blue (B) and red (R) light: area above 3, 8, 20 sigmas: blue disk extent vs red bulge extent.

    python galaxy/editor/v8.py build | knots <acts layer 8> | lda <acts layer 14> | qda <acts layer 14> [options]
"""
import argparse

import numpy as np

import v7
import v6
import v5
import v5b

M = 27
c = np.arange(M) - M // 2
v5.M = M
v5.c = c
v5.Y, v5.X = np.meshgrid(c, c, indexing="ij")
v5.R = np.hypot(v5.X, v5.Y)
v5.TH = np.arctan2(v5.Y, v5.X)
v5.CENTRE = np.zeros((M, M))
v5.CENTRE[M // 2, M // 2] = 1.0
v5b.CENTRE = v5.CENTRE
v6.KCONV = 13

STATS = {"HPc": (0.0, 0.0033), "B": (0.1263, 0.0216), "R": (0.1311, 0.0285)}   # pass 1: (sky level, noise)
AREA_LEVELS = {"HPc": [3, 6, 12], "B": [3, 8, 20], "R": [3, 8, 20]}


def filters():
    f = v7.filters()
    base = {k: v for k, v in f.items()}
    for ch, levels in AREA_LEVELS.items():
        kern = base[ch][0]
        sky, sig = STATS[ch]
        for n in levels:
            t = sky + n * sig
            f[f"{ch}area{n}_a"] = (kern / sig, -t / sig)
            f[f"{ch}area{n}_b"] = (kern / sig, -t / sig - 1)
    out = {}
    for name, v in f.items():
        k = np.zeros((3, 13, 13))
        k[:, 2:11, 2:11] = v[0]                    # same filter, zero-padded to 13x13
        out[name] = (k,) + tuple(v[1:])
    return out


def layout():
    rows, feats = v7.layout()
    D = v5.disk(v5.RMAX)
    for ch, levels in AREA_LEVELS.items():
        for n in levels:
            rows.append((f"{ch}area{n}_a", D))
            rows.append((f"{ch}area{n}_b", D))
            feats.append((f"{ch}_area{n}", {len(rows) - 2: 1.0, len(rows) - 1: -1.0}))
    return rows, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "lda", "qda"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--nq", type=int, default=9)
    ap.add_argument("--out", default="v8_s3")
    a = ap.parse_args()
    if a.stage == "build":
        v6.build("v8", layout, filters)
    elif a.stage == "knots":
        v6.knots(a.acts, "v8")
    elif a.stage == "lda":
        v6.lda(a.acts, a.temp, a.shrink, a.out, "v8")
    else:
        v6.qda(a.acts, a.temp, a.alpha, a.shrink, a.out, a.nq, 0.0, "v8")
