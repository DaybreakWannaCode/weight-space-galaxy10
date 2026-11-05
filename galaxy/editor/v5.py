"""Programmed network v5, written by the editor (Claude): v4's measurements packed under the 500,000-number limit.

v1-v4 put every measurement in one dense Linear over all channels at every pixel (25.7M numbers, 2% nonzero).
v5 gives each measurement ("row") its own channel and its own weight map:

    0  AvgPool2d(4)                                    256 -> 64 px
    1  Conv2d(3 -> F1, 9x9, no padding)                one filter per row: L, R, B (sky-subtracted), fine/coarse
                                                       band-pass, edge direction, or a bar-scale ring m=2 kernel
    2  ReLU   3 AvgPool2d(2)                           56 -> 28 px (32-px units; the 4 px border is cropped)
    4  Conv2d(F1 -> F1, 28x28, groups=F1)              each row's weight map on its own channel: 784 weights a row
    5  ReLU   6 Conv2d(F1 -> F2, 1x1)                  rows -> features (sums of positive parts, |v| from 8 dirs)
    7  ReLU   8 Conv2d(F2 -> F2*K, groups=F2)  9 ReLU  10 Conv2d(F2*K -> F2, groups=F2)     piecewise-linear log
    11 Conv2d(F2 -> F2*K2, groups=F2)  12 ReLU  13 Conv2d(F2*K2 -> 2*F2, groups=F2)       u, u^2 (interleaved)
    14 Flatten  15 Linear(2*F2 -> 10)                  LDA head (closed form)

Thresholds are reused from passes 1 and 13 (the same filters). Bar-scale rings (radius 1.5-2.5 and 2.5-4 px at
64 px) are computed by the 9x9 kernels at full resolution and read at the centre.

    python galaxy/editor/v5.py build                    -> nets/v5_s1
    python galaxy/editor/v5.py knots <acts layer 7>     -> nets/v5_s2
    python galaxy/editor/v5.py head  <acts layer 14> [--quad 0|1] [--temp T] [--shrink S] --out NAME
"""
import argparse
import json
from pathlib import Path

import numpy as np

import v2

NDIR = 8
PHIS = 2 * np.pi * np.arange(NDIR) / NDIR
K, K2 = v2.K, v2.K2
M = 28
c = np.arange(M) + 0.5 - M / 2                       # 32-px units (1 unit = 8 original px)
Y, X = np.meshgrid(c, c, indexing="ij")
R = np.hypot(X, Y)
TH = np.arctan2(Y, X)
RMAX = 13.5
CENTRE = np.zeros((M, M))
CENTRE[M // 2 - 1:M // 2 + 1, M // 2 - 1:M // 2 + 1] = 0.25

_a = np.arange(9) - 4
KY, KX = np.meshgrid(_a, _a, indexing="ij")         # kernel offsets, 64-px units
KR, KTH = np.hypot(KX, KY), np.arctan2(KY, KX)


def disk(r):
    return np.clip(r + 0.5 - R, 0, 1)


def ring(rr, a, b):
    return np.clip(np.minimum(rr - a + 0.5, b + 0.5 - rr), 0, 1)


def gwin(s):
    return np.exp(-R ** 2 / (2 * s ** 2)) * disk(RMAX)


def mom(w, p):
    return w * ((X ** 2 - Y ** 2) * np.cos(p) + 2 * X * Y * np.sin(p))


def gauss(s):
    g = np.exp(-(KX ** 2 + KY ** 2) / (2 * s ** 2))
    return g / g.sum()


def pass_biases():
    """First-conv biases set in passes 1 and 13: (L, R, B, HPf, HPc) and (D0, D60, D120). Read from those passes'
    networks when present, else from pass_stats.json, which holds the same numbers."""
    try:
        _, s1, _ = v2.load_net("v1_s2")
        _, s4, _ = v2.load_net("v4_s1")
        return list(s1["1.bias"][:5]), list(s4["1.bias"][5:8])
    except FileNotFoundError:
        st = json.loads(Path(__file__).with_name("pass_stats.json").read_text())
        return st["v1_s2_conv_bias"], st["v4_s1_conv_bias_5_8"]


def filters():
    """name -> (kernel (3, 9, 9), bias). Biases from the sky/noise statistics of passes 1 and 13."""
    b5, d3 = pass_biases()
    b = dict(zip(["L", "R", "B", "HPf", "HPc"], b5))
    f = {}
    k = np.zeros((3, 9, 9))
    k[:, 4, 4] = 1 / 3
    f["L"] = (k, b["L"])
    k = np.zeros((3, 9, 9))
    k[0, 4, 4] = 1
    f["R"] = (k, b["R"])
    k = np.zeros((3, 9, 9))
    k[2, 4, 4] = 1
    f["B"] = (k, b["B"])
    delta = np.zeros((9, 9))
    delta[4, 4] = 1
    f["HPf"] = (np.repeat((delta - gauss(1.0))[None] / 3, 3, 0), b["HPf"])
    f["HPc"] = (np.repeat((gauss(1.0) - gauss(2.0))[None] / 3, 3, 0), b["HPc"])
    g = np.exp(-(KX ** 2 + KY ** 2) / 2.0)
    g /= g.sum()
    for i, d in enumerate([0, 60, 120]):
        ker = (KX * np.cos(np.radians(d)) + KY * np.sin(np.radians(d))) * g
        f[f"D{d}"] = (np.repeat(ker[None] / 3, 3, 0), d3[i])
    rA, rB = ring(KR, 1.5, 2.5), ring(KR, 2.5, 4.0)
    for j, p in enumerate(PHIS):
        a_, b_ = rA * np.cos(2 * KTH - p), rB * np.cos(2 * KTH - p)
        f[f"ringA_{j}"] = (np.repeat(a_[None] / 3, 3, 0), 0.0)
        f[f"ringB_{j}"] = (np.repeat(b_[None] / 3, 3, 0), 0.0)
        f[f"turnAB+_{j}"] = (np.repeat((a_ + b_)[None] / 3, 3, 0), 0.0)
        f[f"turnAB-_{j}"] = (np.repeat((a_ - b_)[None] / 3, 3, 0), 0.0)
    return f


def pairs(n, rmax, block):
    """180-degree partner cells on an n x n grid of (block x block) cells of the 28 px map, within rmax."""
    cc = (np.arange(n) + 0.5 - n / 2) * block
    out = []
    for i in range(n):
        for j in range(n):
            pi, pj = n - 1 - i, n - 1 - j
            if (i, j) >= (pi, pj) or np.hypot(cc[i], cc[j]) > rmax:
                continue
            a, b = np.zeros((M, M)), np.zeros((M, M))
            a[i * block:(i + 1) * block, j * block:(j + 1) * block] = 1
            b[pi * block:(pi + 1) * block, pj * block:(pj + 1) * block] = 1
            out.append((a, b))
    return out


def layout():
    rows, feats = [], []          # rows: (filter name, 28x28 map); feats: (name, {row: coef})

    def row(filt, w):
        rows.append((filt, w))
        return len(rows) - 1

    def mag(name, filt, maps):    # |v| from the positive parts of its projections on 8 directions
        feats.append((name, {row(filt, m): 1 / NDIR for m in maps}))

    for r in [1.5, 2.5, 3.5, 5, 7, 10, 13.5]:
        feats.append((f"L_disk{r}", {row("L", disk(r)): 1.0}))
    for ch in ("R", "B"):
        for r in [2.5, 5, 10]:
            feats.append((f"{ch}_disk{r}", {row(ch, disk(r)): 1.0}))
    for ch in ("HPf", "HPc"):
        for r in [4, 8, 13.5]:
            feats.append((f"{ch}_disk{r}", {row(ch, disk(r)): 1.0}))
    for s in [1.5, 2.5, 4, 6, 9]:
        mag(f"elong_s{s}", "L", [mom(gwin(s), p) for p in PHIS])
        feats.append((f"size_s{s}", {row("L", gwin(s) * R ** 2): 1.0}))
        feats.append((f"flux_s{s}", {row("L", gwin(s)): 1.0}))
    for sa, sb, cc, sign in [(1.5, 4, 0.375 ** 4, 1), (1.5, 4, 0.375 ** 4, -1), (1.5, 4, 0.375 ** 2, 1),
                             (2.5, 6, (5 / 12) ** 4, 1)]:
        mag(f"align_{sa}_{sb}_c{cc:.3f}_{'plus' if sign > 0 else 'minus'}", "L",
            [mom(gwin(sa), p) + sign * cc * mom(gwin(sb), p) for p in PHIS])
    wsp = np.clip((R - 1) / 1, 0, 1) * (1 - np.clip((R - 9) / 4, 0, 1))
    for filt, w, name in [("L", gwin(4), "m1_L_s4"), ("L", gwin(9), "m1_L_s9"), ("HPc", wsp, "m1_HPc")]:
        mag(name, filt, [w * (X * np.cos(p) + Y * np.sin(p)) for p in PHIS])
    edges = [2.25, 3.375, 5.06, 7.59, 11.39]
    rings = [ring(R, edges[j], edges[j + 1]) for j in range(4)]
    m2 = [[w * np.cos(2 * TH - p) for p in PHIS] for w in rings]
    for j in range(4):
        mag(f"ring{j + 2}_m2", "L", m2[j])
        feats.append((f"ring{j + 2}_flux", {row("L", rings[j]): 1.0}))
    for j in range(3):
        for sign, sn in ((1, "plus"), (-1, "minus")):
            mag(f"turn{j + 2}_{j + 3}_{sn}", "L", [m2[j][k] + sign * m2[j + 1][k] for k in range(NDIR)])
    for a, b in [(1.5, 4), (4, 7.5), (7.5, 12)]:
        w = ring(R, a, b)
        for kind, fn in (("rad", np.cos), ("tan", np.sin)):
            feats.append((f"edge_{kind}_{a}_{b}",
                          {row(f"D{d}", w * fn(np.radians(d) - TH) ** 2): 1.0 for d in (0, 60, 120)}))
    for filt, n, block, rmax, name in [("L", M, 1, 3.5, "asym_L_inner"), ("L", 7, 4, RMAX, "asym_L"),
                                       ("HPc", 7, 4, RMAX, "asym_HPc")]:
        coefs = {}
        for a, b in pairs(n, rmax, block):
            coefs[row(filt, a - b)] = 1.0
            coefs[row(filt, b - a)] = 1.0
        feats.append((name, coefs))
    for name, pre in [("ringA_m2", "ringA_"), ("ringB_m2", "ringB_"), ("turnAB_plus", "turnAB+_"),
                      ("turnAB_minus", "turnAB-_")]:
        feats.append((name, {row(f"{pre}{j}", CENTRE): 1 / NDIR for j in range(NDIR)}))
    return rows, feats


def arch_for(F1, F2):
    C = lambda i, o, k, g=1, **kw: {"type": "Conv2d", "args": dict(in_channels=i, out_channels=o,  # noqa: E731
                                                                     kernel_size=k, groups=g, **kw)}
    return [
        {"type": "AvgPool2d", "args": {"kernel_size": 4}},
        C(3, F1, 9), {"type": "ReLU"}, {"type": "AvgPool2d", "args": {"kernel_size": 2}},
        C(F1, F1, M, F1), {"type": "ReLU"}, C(F1, F2, 1), {"type": "ReLU"},
        C(F2, F2 * K, 1, F2), {"type": "ReLU"}, C(F2 * K, F2, 1, F2),
        C(F2, F2 * K2, 1, F2), {"type": "ReLU"}, C(F2 * K2, 2 * F2, 1, F2),
        {"type": "Flatten"}, {"type": "Linear", "args": {"in_features": 2 * F2, "out_features": 10}},
    ]


def build(prefix="v5", filters_fn=None, layout_fn=None):
    rows, feats = (layout_fn or layout)()
    F1, F2 = len(rows), len(feats)
    flt = (filters_fn or filters)()
    # a filter may carry a third entry, an offset added by its conv bias so the first ReLU never clips it; the
    # row subtracts it again (offset x sum of the row's weight map), so the next ReLU acts on the read-out value
    off = np.array([flt[f][2] * w.sum() if len(flt[f]) > 2 else 0.0 for f, w in rows])
    sd = {"1.weight": np.stack([flt[f][0] for f, _ in rows]), "1.bias": np.array([flt[f][1] for f, _ in rows]),
          "4.weight": np.stack([w for _, w in rows])[:, None], "4.bias": -off}
    w6 = np.zeros((F2, F1))
    for i, (_, coefs) in enumerate(feats):
        for r, v in coefs.items():
            w6[i, r] = v
    sd["6.weight"], sd["6.bias"] = w6[:, :, None, None], np.zeros(F2)
    for k, shape in {"8": (F2 * K, 1), "10": (F2, K), "11": (F2 * K2, 1), "13": (2 * F2, K2)}.items():
        sd[f"{k}.weight"], sd[f"{k}.bias"] = np.zeros(shape + (1, 1)), np.zeros(shape[0])
    sd["15.weight"], sd["15.bias"] = np.zeros((10, 2 * F2)), np.zeros(10)
    v2.save(f"{prefix}_s1", arch_for(F1, F2), sd, dict(features=[n for n, _ in feats], F1=F1, F2=F2,
                                                rows=[f for f, _ in rows]))


def knots(acts_file, prefix="v5"):
    arch, sd, meta = v2.load_net(f"{prefix}_s1")
    f = np.load(acts_file)["acts"]
    F2 = f.shape[1]
    tq = np.arange(K2) / K2
    sq = tq + np.append(tq[1:], 1.0)
    aq = np.concatenate([[sq[0]], np.diff(sq)])
    w8, b8, w10, b10 = np.zeros((F2 * K, 1, 1, 1)), np.zeros(F2 * K), np.zeros((F2, K, 1, 1)), np.zeros(F2)
    w11, b11, w13, b13 = np.zeros((F2 * K2, 1, 1, 1)), np.zeros(F2 * K2), np.zeros((2 * F2, K2, 1, 1)), np.zeros(2 * F2)
    info = {}
    for i in range(F2):
        pos = f[:, i][f[:, i] > 0]
        lo = np.quantile(pos, 0.005) if len(pos) else 1e-3
        hi = max(np.quantile(f[:, i], 0.995), lo * 4)
        t = np.geomspace(lo, hi, K)
        v = np.log(t)
        s = np.diff(v) / np.diff(t)
        w8[i * K:(i + 1) * K, 0, 0, 0], b8[i * K:(i + 1) * K] = 1.0, -t
        w10[i, :, 0, 0], b10[i] = np.concatenate([[s[0]], np.diff(s), [-s[-1]]]), v[0]
        a_, b_ = v[0], v[-1]
        w11[i * K2:(i + 1) * K2, 0, 0, 0] = 1 / (b_ - a_)
        b11[i * K2:(i + 1) * K2] = -a_ / (b_ - a_) - tq
        w13[2 * i, 0, 0, 0] = 1.0                    # u (the knot at 0)
        w13[2 * i + 1, :, 0, 0] = aq                 # piecewise-linear u^2
        info[meta["features"][i]] = dict(lo=float(lo), hi=float(hi), share_zero=float((f[:, i] <= 0).mean()))
    sd.update({"8.weight": w8, "8.bias": b8, "10.weight": w10, "10.bias": b10,
               "11.weight": w11, "11.bias": b11, "13.weight": w13, "13.bias": b13})
    meta["knots"] = info
    bad = [k for k, v in info.items() if v["share_zero"] > 0.2]
    print(f"{F2} features; zero for >20% of galaxies: {bad or 'none'}")
    v2.save(f"{prefix}_s2", arch, sd, meta)


def head(acts_file, quad, temp, shrink, out, prefix="v5"):
    arch, sd, meta = v2.load_net(f"{prefix}_s2")
    zf = np.load(acts_file)
    z, y = zf["acts"].astype(np.float64), zf["labels"]
    cols = np.arange(z.shape[1]) if quad else np.arange(0, z.shape[1], 2)
    z = z[:, cols]
    C = 10
    prior = np.bincount(y, minlength=C) / len(y)
    mu = np.stack([z[y == k].mean(0) for k in range(C)])
    res = z - mu[y]
    S = res.T @ res / (len(y) - C)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    S += 1e-6 * np.trace(S) / len(S) * np.eye(len(S))
    W = mu @ np.linalg.inv(S)
    w15 = np.zeros((C, zf["acts"].shape[1]))
    w15[:, cols] = temp * W
    sd["15.weight"], sd["15.bias"] = w15, temp * (-0.5 * np.einsum("cf,cf->c", W, mu)) + np.log(prior)
    meta["head"] = dict(quad=bool(quad), temp=temp, shrink=shrink, acts=str(acts_file))
    v2.save(out, arch, sd, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "head"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--quad", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v5_s3")
    a = ap.parse_args()
    {"build": build, "knots": lambda: knots(a.acts),
     "head": lambda: head(a.acts, a.quad, a.temp, a.shrink, a.out)}[a.stage]()
