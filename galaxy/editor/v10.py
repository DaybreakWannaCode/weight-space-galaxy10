"""Programmed network v10, written by the editor (Claude): v9 repacked, and the freed room spent on radial profiles of
colour and fine structure.

Repack (same measurements, ~150k numbers freed):
  1. |v| of a 2-vector from 5 directions instead of 8. With an odd count no direction is another's opposite, so
     sum_j relu(v . e_j) = sum_j |v . e_j| / 2 over 10 half-directions 36 degrees apart: the rotation ripple is
     1 / cos 18 = 5.1%, against 1 / cos 22.5 = 8.2% for 8 directions. 3 channels saved per magnitude (32 of them),
     and 15 first-conv filters saved for the centre-read bar measurements.
  2. asymmetry sums: |x| = 2 relu(x) - x, so sum_p |x_p| = 2 sum_p relu(x_p) - relu(S) + relu(-S), S = sum_p x_p:
     one channel per cell pair plus two, instead of two per pair (asym_L_inner, asym_L, shape_asym).

New (diagnosed on the pass-45 training activations: for tight / loose the strongest single measurements are central
fine structure HPf_disk4, red light and size; for disturbed / in-between, inner asymmetry and blue light):
  - radial profiles in 5 annuli (0-1.5, 1.5-3, 3-5, 5-8, 8-12 map units) of: fine band-pass HPf on L, R and B;
    coarse band-pass HPc on L, R and B; R and B light (colour gradients, the colour of clumps and arms);
  - HPf area above 6 and 12 noise sigmas (how much of the galaxy is clumpy, not how bright the clumps are);
  - m = 1, 2, 3 of HPc in a core annulus (1 - 2.5): the inner-ring modes were v8's strongest HPc modes;
  - inner (r < 3.5) pixel-pair asymmetry of the blue light and of HPc; m = 1 of the blue light at r 2-5 and 5-10.
Per-band noise (pass 1: L 0.0144, R 0.0285, B 0.0216 per 64-px pixel) sets the band-passed filters' biases at 3 sigma
of their noise, scaled from the L versions (HPf -0.0345, HPc -0.0100).

    python galaxy/editor/v10.py build | knots <acts layer 8> | lda <acts layer 14> | qda <acts layer 14> [options]
"""
import argparse

import numpy as np

import v9
import v6
import v5
import v5b
import v2

NDIR = 5
for mod in (v5, v5b):
    mod.NDIR = NDIR
    mod.PHIS = 2 * np.pi * np.arange(NDIR) / NDIR

ASYM = ("asym_L_inner", "asym_L", "shape_asym")
EDGES = [0, 1.5, 3, 5, 8, 12]
NOISE = {"L": 0.0144, "R": 0.0285, "B": 0.0216}
SIG_HPF, SIG_HPC = 0.0115, 0.0033          # L versions (biases -0.0345, -0.0100 = 3 sigma)


def pad13(k9):
    k = np.zeros(k9.shape[:-2] + (13, 13))
    k[..., 2:11, 2:11] = k9
    return k


def filters():
    f = v9.filters()
    delta = np.zeros((9, 9))
    delta[4, 4] = 1
    hpf, hpc = delta - v5.gauss(1.0), v5.gauss(1.0) - v5.gauss(2.0)
    for band, ch in (("R", 0), ("B", 2)):
        scale = NOISE[band] / NOISE["L"]
        for name, ker, sig in (("HPf", hpf, SIG_HPF), ("HPc", hpc, SIG_HPC)):
            k = np.zeros((3, 9, 9))
            k[ch] = ker
            f[f"{name}_{band}"] = (pad13(k), -3 * sig * scale)
    k_hpf = pad13(np.repeat(hpf[None] / 3, 3, 0))
    for n in (6, 12):
        kk = 1 / SIG_HPF
        f[f"HPfarea{n}_a"] = (k_hpf * kk, -kk * n * SIG_HPF)
        f[f"HPfarea{n}_b"] = (k_hpf * kk, -kk * n * SIG_HPF - 1)
    return f


def annuli():
    R = v5.R
    return [v5.disk(EDGES[1])] + [v5.ring(R, EDGES[i], EDGES[i + 1]) for i in range(1, len(EDGES) - 1)]


def inner_asym(spec, name, rows, feats):
    pr = v5.pairs(v5.M, 3.5, 1)
    coefs, S = {}, 0
    for a, b in pr:
        rows.append((spec, a - b))
        coefs[len(rows) - 1] = 2.0
        S = S + (a - b)
    rows.append((spec, S))
    coefs[len(rows) - 1] = -1.0
    rows.append((spec, -S))
    coefs[len(rows) - 1] = 1.0
    feats.append((name, coefs))


def layout():
    old_rows, old_feats = v9.layout()
    rows, feats = [], []
    for name, coefs in old_feats:
        rs = sorted(coefs)
        if name not in ASYM:
            new = {}
            for r in rs:
                rows.append(old_rows[r])
                new[len(rows) - 1] = coefs[r]
            feats.append((name, new))
            continue
        spec = old_rows[rs[0]][0]
        new, S = {}, 0
        for r1, r2 in zip(rs[0::2], rs[1::2]):
            m1, m2 = old_rows[r1][1], old_rows[r2][1]
            assert np.allclose(m1, -m2) and old_rows[r1][0] == spec and old_rows[r2][0] == spec
            rows.append((spec, m1))
            new[len(rows) - 1] = 2.0
            S = S + m1
        rows.append((spec, S))
        new[len(rows) - 1] = -1.0
        rows.append((spec, -S))
        new[len(rows) - 1] = 1.0
        feats.append((name, new))
    for filt in ("HPf", "HPf_R", "HPf_B", "HPc", "HPc_R", "HPc_B", "R", "B"):
        for i, w in enumerate(annuli()):
            rows.append((filt, w))
            feats.append((f"{filt}_ann{EDGES[i]}-{EDGES[i + 1]}", {len(rows) - 1: 1.0}))
    D = v5.disk(v5.RMAX)
    for n in (6, 12):
        rows.append((f"HPfarea{n}_a", D))
        rows.append((f"HPfarea{n}_b", D))
        feats.append((f"HPf_area{n}", {len(rows) - 2: 1.0, len(rows) - 1: -1.0}))
    w = v5.ring(v5.R, 1.0, 2.5)
    for m in (1, 2, 3):
        idx = []
        for p in v5.PHIS:
            rows.append(("HPc", w * np.cos(m * v5.TH - p)))
            idx.append(len(rows) - 1)
        feats.append((f"hpc_core_m{m}", {r: 1 / NDIR for r in idx}))
    for nm, (a, b) in (("in", (2, 5)), ("out", (5, 10))):          # lopsidedness of the blue light
        w = v5.ring(v5.R, a, b)
        idx = []
        for p in v5.PHIS:
            rows.append(("B", w * (v5.X * np.cos(p) + v5.Y * np.sin(p))))
            idx.append(len(rows) - 1)
        feats.append((f"m1_B_{nm}", {r: 1 / NDIR for r in idx}))
    inner_asym("B", "asym_B_inner", rows, feats)
    inner_asym("HPc", "asym_HPc_inner", rows, feats)
    return rows, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "lda", "qda"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--nq", type=int, default=9)
    ap.add_argument("--out", default="v10_s3")
    a = ap.parse_args()
    if a.stage == "build":
        v9.build("v10", layout, filters)
    elif a.stage == "knots":
        v6.knots(a.acts, "v10")
    elif a.stage == "lda":
        v6.lda(a.acts, a.temp, a.shrink, a.out, "v10")
    else:
        v6.qda(a.acts, a.temp, a.alpha, a.shrink, a.out, a.nq, 0.0, "v10")
