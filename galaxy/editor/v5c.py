"""Programmed network v5c, written by the editor (Claude): v5b with the bar kernels read exactly at the centre.

On v5/v5b's 28x28 map the galaxy centre falls between pooled pixels, so "the centre" averaged ring kernels centred up
to 1.5 px (64 px units) away, smearing rings of radius 2-3 px: barred-vs-tight separation 1.37 sd vs 1.73 in v4.
v5c pools with padding=1, giving a 29x29 map whose middle pixel (14, 14) is centred on the galaxy (it averages the 4
conv outputs at +-0.5 px). Coarse rows use that grid's integer coordinates; the 4x4 asymmetry cells are re-tiled
symmetrically about the centre (the centre row and column are left out).

    python galaxy/editor/v5c.py build | knots <acts layer 7> | head <acts layer 14> [--quad] [--temp] [--shrink] --out
"""
import argparse

import numpy as np

import v5
import v5b

M = 29
c = np.arange(M) - M // 2                            # integer coordinates, 32-px units; (14, 14) is the centre
v5.M = M
v5.c = c
v5.Y, v5.X = np.meshgrid(c, c, indexing="ij")
v5.R = np.hypot(v5.X, v5.Y)
v5.TH = np.arctan2(v5.Y, v5.X)
v5.CENTRE = np.zeros((M, M))
v5.CENTRE[M // 2, M // 2] = 1.0
v5b.CENTRE = v5.CENTRE

_orig_pairs = v5.pairs


def pairs(n, rmax, block):
    """Pixel pairs (block 1) as before; 4x4 cells tiled symmetrically: x in [1,4], [5,8], ... and mirrored."""
    if block == 1:
        return _orig_pairs(n, rmax, block)
    cell = lambda v: (np.sign(v) * np.ceil(np.abs(v) / block)).astype(int)   # noqa: E731; 0 for the centre line
    cy, cx = cell(v5.Y), cell(v5.X)
    out = []
    for by in range(-4, 5):
        for bx in range(-4, 5):
            if by == 0 or bx == 0 or (by, bx) <= (-by, -bx):
                continue
            a = ((cy == by) & (cx == bx)).astype(float)
            b = ((cy == -by) & (cx == -bx)).astype(float)
            if not a.any():
                continue
            centre = np.hypot((v5.Y * a).sum() / a.sum(), (v5.X * a).sum() / a.sum())
            if centre <= rmax:
                out.append((a, b))
    return out


v5.pairs = pairs
_orig_arch = v5.arch_for


def arch_for(F1, F2):
    arch = _orig_arch(F1, F2)
    arch[3]["args"]["padding"] = 1                    # AvgPool2d(2, padding=1): 56 -> 29 px
    arch[4]["args"]["kernel_size"] = M
    return arch


v5.arch_for = arch_for

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "head"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--quad", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v5c_s3")
    a = ap.parse_args()
    if a.stage == "build":
        v5.build("v5c", v5b.filters, v5b.layout)
    elif a.stage == "knots":
        v5.knots(a.acts, "v5c")
    else:
        v5.head(a.acts, a.quad, a.temp, a.shrink, a.out, "v5c")
