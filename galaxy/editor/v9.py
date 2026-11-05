"""Programmed network v9, written by the editor (Claude): v8 with spiral pitch-angle measurements.

Diagnosis (v8 val confusion + class-pair Mahalanobis distances on the pass-40 training activations): most of the error
is among spirals. Tight -> loose 48, loose -> tight 24, disturbed -> loose 32, barred -> loose 25; the weakest pairs are
disturbed/loose 1.48 and tight/loose 1.84. Tight vs loose is, by definition, the arms' pitch angle, which v8 only
sees through ring-to-ring m=2 alignments (turn features), and those wrap: between rings 1.5x apart the m=2 phase
turns by ~4.6 rad for a 10-degree spiral and ~1.4 rad for a 30-degree one.

New: log-spiral amplitudes (the Fourier transform in log-polar coordinates used to measure pitch angles),
    A(2, p) = sum over pixels of HPc(x) w(r) exp(-i (2 theta + p ln r)),   tan(pitch) = 2 / p,
for p = 2.5, 4.5, 7.5, 12 (pitch 39, 24, 15, 9.5 degrees) on the coarse band-pass map (the arms), with w = annulus / r.
|A| comes from the positive parts of 8 phase projections (the |v| trick); a galaxy winds one way or the other, so each
feature is |A(2, p)| + |A(2, -p)|: 16 channels. The inner edge of each annulus keeps the template's wavelength above
2 map pixels.

Also new, for disturbed galaxies (the weakest pair is disturbed / loose): shape asymmetry (Pawlik et al. 2016), the
180-degree asymmetry of a deep detection mask, which reaches faint tidal features that the light-weighted asymmetry
does not: M = clamp((L_smooth - sky - 1.5 sigma) / 0.5 sigma, 0, 1) from two first-conv filters (Gaussian-smoothed
luminance), formed in the copy layer; sum |M(x) - M(-x)| over the 20 pairs of 4x4 cells, and the mask's area.

Paid for by (all from the pass-40 training activations, closed-form):
  - the five centre-read bar measurements (ringA, ringB, turnAB+-, momA): their 8 projections are now clipped by the
    first ReLU (no +25 offset) and summed by the copy layer into one channel each: 40 channels -> 5;
  - dropping asym_HPc, turn3_4_plus, turn4_5_plus, elong_s6, align_2.5_6 (72 channels): removing them together
    shrinks the class-pair Mahalanobis distances by 0.17% on average.

    python galaxy/editor/v9.py build | knots <acts layer 8> | lda <acts layer 14> | qda <acts layer 14> [options]
"""
import argparse

import numpy as np

import v8
import v6
import v5
import v2

FINE = {"ringA_m2": "ringA_", "ringB_m2": "ringB_", "turnAB_plus": "turnAB+_", "turnAB_minus": "turnAB-_",
        "elong_fine_s2": "momA_"}
DROP = {"asym_HPc", "turn3_4_plus", "turn4_5_plus", "elong_s6", "align_2.5_6_c0.030_plus"}
LSP = [(2.5, 2.0, 11.0), (4.5, 2.0, 11.0), (7.5, 3.0, 11.0), (12.0, 5.0, 11.0)]   # (p, inner, outer radius)
SKY, SIG = 0.1277, 0.0144            # pass 1: sky level and noise of the 4x-pooled luminance
MASK_T, MASK_W = SKY + 1.5 * SIG, 0.5 * SIG      # deep mask: smoothed luminance above 1.5 sigma (ramp 0.5 sigma)


def filters():
    f = v8.filters()
    for pre in FINE.values():
        for j in range(v5.NDIR):
            f[f"{pre}{j}"] = (f[f"{pre}{j}"][0], 0.0)     # no offset: the first ReLU takes the positive part
    a = np.arange(13) - 6
    yy, xx = np.meshgrid(a, a, indexing="ij")
    g = np.exp(-(xx ** 2 + yy ** 2) / 2.0)
    g = np.repeat((g / g.sum())[None] / 3, 3, 0)            # luminance smoothed by a sigma = 1 px Gaussian
    k = 1 / MASK_W
    f["Lsm_a"] = (g * k, -k * MASK_T)
    f["Lsm_b"] = (g * k, -k * MASK_T - 1)                  # mask = relu(k (L - t)) - relu(k (L - t) - 1)
    return f


def layout():
    old_rows, old_feats = v8.layout()
    rows, feats = [], []
    for name, coefs in old_feats:
        if name in DROP:
            continue
        if name in FINE:
            spec = {old_rows[r][0]: 1 / v5.NDIR for r in coefs}          # sum of the 8 clipped projections
            rows.append((spec, v5.CENTRE))
            feats.append((name, {len(rows) - 1: 1.0}))
            continue
        new = {}
        for r, v in coefs.items():
            rows.append(old_rows[r])
            new[len(rows) - 1] = v
        feats.append((name, new))
    R, TH = v5.R, v5.TH
    lnr = np.log(np.maximum(R, 0.5))
    for p, a, b in LSP:
        w = v5.ring(R, a, b) / np.maximum(R, 1.0)
        idx = []
        for s in (1, -1):
            for ph in v5.PHIS:
                rows.append(("HPc", w * np.cos(2 * TH + s * p * lnr - ph)))
                idx.append(len(rows) - 1)
        feats.append((f"lsp_p{p}", {r: 1 / v5.NDIR for r in idx}))
    mask = {"Lsm_a": 1.0, "Lsm_b": -1.0}                    # the copy layer forms the soft detection mask
    rows.append((mask, v5.disk(v5.RMAX)))
    feats.append(("mask_area", {len(rows) - 1: 1.0}))
    coefs = {}
    for a, b in v5.pairs(7, v5.RMAX, 4):                    # shape asymmetry: sum |M(x) - M(-x)| over 4x4 cells
        rows.append((mask, a - b))
        coefs[len(rows) - 1] = 1.0
        rows.append((mask, b - a))
        coefs[len(rows) - 1] = 1.0
    feats.append(("shape_asym", coefs))
    return rows, feats


def build(prefix="v9", layout_fn=None, filters_fn=None):
    rows, feats = (layout_fn or layout)()
    F1, F2 = len(rows), len(feats)
    flt = (filters_fn or filters)()
    used = set()
    for spec, _ in rows:
        used.update(spec if isinstance(spec, dict) else [spec])
    base = [f for f in flt if f in used]
    B = len(base)
    sd = {"1.weight": np.stack([flt[f][0] for f in base]), "1.bias": np.array([flt[f][1] for f in base])}
    copy = np.zeros((F1, B, 1, 1))
    off = np.zeros(F1)
    for r, (spec, w) in enumerate(rows):
        if isinstance(spec, dict):
            for f, v in spec.items():
                copy[r, base.index(f)] = v
                assert len(flt[f]) == 2
        else:
            copy[r, base.index(spec)] = 1.0
            if len(flt[spec]) > 2:
                off[r] = flt[spec][2] * w.sum()
    sd["4.weight"], sd["4.bias"] = copy, np.zeros(F1)
    sd["5.weight"], sd["5.bias"] = np.stack([w for _, w in rows])[:, None], -off
    w7 = np.zeros((F2, F1, 1, 1))
    for i, (_, coefs) in enumerate(feats):
        for r, v in coefs.items():
            w7[i, r] = v
    sd["7.weight"], sd["7.bias"] = w7, np.zeros(F2)
    K, K2 = v6.K, v6.K2
    for k, shape in {"9": (F2 * K, 1), "11": (F2, K), "12": (F2 * K2, 1), "14": (2 * F2, K2)}.items():
        sd[f"{k}.weight"], sd[f"{k}.bias"] = np.zeros(shape + (1, 1)), np.zeros(shape[0])
    sd["16.weight"], sd["16.bias"] = np.zeros((10, 2 * F2)), np.zeros(10)
    v2.save(f"{prefix}_s1", v6.arch_lda(B, F1, F2), sd, dict(features=[n for n, _ in feats], F1=F1, F2=F2, B=B,
                                                               filters=base))
    G = 9 + 36 + 9
    qda_extra = 9 * (2 * F2 + 1) + G * 16 * (9 + 1) + G * (16 + 1) + 10 * (G + 1) - 10 * (2 * F2 + 1)
    print(f"B {B}  F1 {F1}  F2 {F2}  LDA net {sum(v.size for v in sd.values())}  "
          f"QDA net {sum(v.size for v in sd.values()) + qda_extra}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "lda", "qda"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--nq", type=int, default=9)
    ap.add_argument("--out", default="v9_s3")
    a = ap.parse_args()
    if a.stage == "build":
        build()
    elif a.stage == "knots":
        v6.knots(a.acts, "v9")
    elif a.stage == "lda":
        v6.lda(a.acts, a.temp, a.shrink, a.out, "v9")
    else:
        v6.qda(a.acts, a.temp, a.alpha, a.shrink, a.out, a.nq, 0.0, "v9")
