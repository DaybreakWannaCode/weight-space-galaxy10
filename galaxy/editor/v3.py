"""Programmed network v3, written by the editor (Claude): v2 without the band-pass spiral templates (no signal on
train: best pitch-ratio separation of tight vs loose spirals 0.15 sd), plus ring-to-ring winding of the m=2 light.

Winding: in each of 6 rings (radii 2 to 22.5 px, 1.5x apart) the m=2 vector v_j of the sky-subtracted light; a
spiral arm of pitch psi turns v_j by 2 ln(1.5) / tan(psi) between neighbouring rings (3.8 rad at 12 deg, 0.7 at 50,
0 for a bar or a tilted disk). |v_j + c v_k| vs |v_j - c v_k| (c = 1, 3) measures that turn without knowing the
winding sense; rings one and two apart. Also |v_j| and the ring's flux (m=2 strength per ring).

    python galaxy/editor/v3.py build | knots <acts layer 7> | head <acts layer 13> [--quad] [--temp] [--shrink] --out
"""
import argparse

import numpy as np

import v2
from v2 import N_DIR, R, TH, X, Y, block_pairs, disk, gwin, moment_rows  # noqa: F401

EDGES = [2, 3, 4.5, 6.75, 10.125, 15.19, 22.78]


def layout():
    rows, feats = [], []

    def row(name, wd):
        rows.append((name, wd))
        return len(rows) - 1

    def feat(name, coefs):
        feats.append((name, coefs))

    phis = 2 * np.pi * np.arange(N_DIR) / N_DIR
    for r in [3, 5, 7, 10, 14, 20, 27]:
        feat(f"L_disk{r}", {row(f"L_disk{r}", {"L": disk(r)}): 1.0})
    for ch in ("R", "B"):
        for r in [5, 10, 20]:
            feat(f"{ch}_disk{r}", {row(f"{ch}_disk{r}", {ch: disk(r)}): 1.0})
    for ch in ("HPf", "HPc"):
        for r in [8, 16, 27]:
            feat(f"{ch}_disk{r}", {row(f"{ch}_disk{r}", {ch: disk(r)}): 1.0})
    for s in [3, 5, 8, 12, 18]:
        w = gwin(s)
        feat(f"elong_s{s}", {row(f"mom_s{s}_{k}", {"L": moment_rows(w, p)}): 1 / N_DIR for k, p in enumerate(phis)})
        feat(f"size_s{s}", {row(f"size_s{s}", {"L": w * R ** 2}): 1.0})
        feat(f"flux_s{s}", {row(f"flux_s{s}", {"L": w}): 1.0})
    for sa, sb in [(3, 8), (5, 12)]:
        rho = sa / sb
        for cc in [1.0, rho ** 2, rho ** 4]:
            for sign in (+1, -1):
                coefs = {row(f"align_{sa}_{sb}_{cc:.3f}_{sign}_{k}",
                             {"L": moment_rows(gwin(sa), p) + sign * cc * moment_rows(gwin(sb), p)}): 1 / N_DIR
                         for k, p in enumerate(phis)}
                feat(f"align_{sa}_{sb}_c{cc:.3f}_{'plus' if sign > 0 else 'minus'}", coefs)
    wsp = v2.ramp(R, 2, 4) * (1 - v2.ramp(R, 18, 26))
    for ch, w, name in [("L", gwin(8), "m1_L_s8"), ("L", gwin(18), "m1_L_s18"), ("HPc", wsp, "m1_HPc")]:
        feat(name, {row(f"{name}_{k}", {ch: w * (X * np.cos(p) + Y * np.sin(p))}): 1 / N_DIR
                    for k, p in enumerate(phis)})
    for ch, block, rmax, name in [("L", 2, 10, "asym_L_inner"), ("L", 4, v2.RMAX, "asym_L"),
                                  ("HPc", 4, v2.RMAX, "asym_HPc")]:
        coefs = {}
        for q, (a, b) in enumerate(block_pairs(block, rmax)):
            coefs[row(f"{name}_{q}+", {ch: a - b})] = 1.0
            coefs[row(f"{name}_{q}-", {ch: b - a})] = 1.0
        feat(name, coefs)
    # ---- new in v3: m=2 per ring and its turn between rings
    rings = []
    for j in range(len(EDGES) - 1):
        a, b = EDGES[j], EDGES[j + 1]
        rings.append(np.clip(np.minimum(R - a + 0.5, b + 0.5 - R), 0, 1))     # soft-edged annulus
    m2 = [[w * np.cos(2 * TH - p) for p in phis] for w in rings]            # v_j projected on direction p
    for j, w in enumerate(rings):
        feat(f"ring{j}_m2", {row(f"ring{j}_m2_{k}", {"L": m2[j][k]}): 1 / N_DIR for k in range(N_DIR)})
        feat(f"ring{j}_flux", {row(f"ring{j}_flux", {"L": w}): 1.0})
    for gap in (1, 2):
        for j in range(len(rings) - gap):
            k2 = j + gap
            for cc in (1.0, 3.0):
                for sign, sn in ((+1, "plus"), (-1, "minus")):
                    coefs = {row(f"turn{j}_{k2}_c{cc}_{sn}_{k}", {"L": m2[j][k] + sign * cc * m2[k2][k]}): 1 / N_DIR
                             for k in range(N_DIR)}
                    feat(f"turn{j}_{k2}_c{cc:g}_{sn}", coefs)
    return rows, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "head"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--quad", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v3_s3")
    a = ap.parse_args()
    if a.stage == "build":
        v2.build(layout, "v3")
    elif a.stage == "knots":
        v2.knots(a.acts, "v3")
    else:
        v2.head(a.acts, a.quad, a.temp, a.shrink, a.out, "v3")
