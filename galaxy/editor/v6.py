"""Programmed network v6, written by the editor (Claude). Builds weights with numpy; never runs the network.

Changes from v5c (diagnosed on training activations: v5c trails v4 by ~3% on every class pair; the weakest pairs are
disturbed/loose 1.41, tight/loose 1.57, barred/loose 1.97 in Mahalanobis distance):
  1. each distinct filter is computed once (48 filters) and copied to the measurement channels by a 1x1 conv,
     instead of one 9x9 filter copy per measurement: ~65k numbers freed;
  2. new measurements: m=1, m=2, m=3 patterns of the coarse band-pass map in an inner and an outer ring
     (two-armed spirals are m=2; irregular light also has m=1 and m=3);
  3. a QDA head: class-specific covariances in the 9 most discriminating (LDA) directions, built in closed form
     from class means and covariances. Its quadratic terms come from ReLU staircases for z_i^2 and (z_i + z_j)^2.

    0 AvgPool2d(4)  1 Conv2d(3->B, 9x9)  2 ReLU  3 AvgPool2d(2, pad 1)  4 Conv2d(B->F1, 1x1) copies
    5 Conv2d(F1->F1, 29x29, groups=F1)  6 ReLU  7 Conv2d(F1->F2, 1x1)  8 ReLU
    9-11 piecewise-linear log  12-14 u, u^2          (as v5, grouped 1x1 convs)
    LDA head: 15 Flatten 16 Linear(2F2 -> 10)
    QDA head: 15 Conv2d(2F2->9) z  16 Conv2d(9->54*16) staircase rows  17 ReLU  18 Conv2d(864->54, groups=54)
              19 Flatten  20 Linear(54 -> 10)

    python galaxy/editor/v6.py build                          -> nets/v6_s1  (LDA architecture, placeholder head)
    python galaxy/editor/v6.py knots <acts layer 8>           -> nets/v6_s2
    python galaxy/editor/v6.py lda <acts layer 14> [--temp] [--shrink] --out NAME
    python galaxy/editor/v6.py qda <acts layer 14> [--temp] [--alpha] [--shrink] --out NAME
"""
import argparse
import itertools

import numpy as np

import v5c  # noqa: F401  (applies the 29x29 centred grid to v5)
import v2
import v5
import v5b

K, K2 = v2.K, v2.K2
KCONV = 9                 # first-conv kernel size
NQ = 9                    # discriminant directions for the QDA head
KS = 8                    # staircase knots per squared quantity (each side)


def layout():
    rows, feats = v5b.layout()
    R, TH = v5.R, v5.TH
    for name, (a, b) in [("in", (2, 5)), ("out", (5, 10))]:
        w = v5.ring(R, a, b)
        for m in (1, 2, 3):
            idx = []
            for p in v5.PHIS:
                rows.append(("HPc", w * np.cos(m * TH - p)))
                idx.append(len(rows) - 1)
            feats.append((f"hpc_{name}_m{m}", {r: 1 / v5.NDIR for r in idx}))
        rows.append(("HPc", w))
        feats.append((f"hpc_{name}_flux", {len(rows) - 1: 1.0}))
    return rows, feats


def C(i, o, k, g=1, **kw):
    return {"type": "Conv2d", "args": dict(in_channels=i, out_channels=o, kernel_size=k, groups=g, **kw)}


def arch_trunk(B, F1, F2):
    return [
        {"type": "AvgPool2d", "args": {"kernel_size": 4}}, C(3, B, KCONV), {"type": "ReLU"},
        {"type": "AvgPool2d", "args": {"kernel_size": 2, "padding": 1}}, C(B, F1, 1),
        C(F1, F1, v5.M, F1), {"type": "ReLU"}, C(F1, F2, 1), {"type": "ReLU"},
        C(F2, F2 * K, 1, F2), {"type": "ReLU"}, C(F2 * K, F2, 1, F2),
        C(F2, F2 * K2, 1, F2), {"type": "ReLU"}, C(F2 * K2, 2 * F2, 1, F2),
    ]


def arch_lda(B, F1, F2):
    return arch_trunk(B, F1, F2) + [{"type": "Flatten"},
                                    {"type": "Linear", "args": {"in_features": 2 * F2, "out_features": 10}}]


def arch_qda(B, F1, F2, nq=NQ):
    G = nq + nq * (nq - 1) // 2 + nq
    return arch_trunk(B, F1, F2) + [C(2 * F2, nq, 1), C(nq, G * 2 * KS, 1), {"type": "ReLU"},
                                    C(G * 2 * KS, G, 1, G), {"type": "Flatten"},
                                    {"type": "Linear", "args": {"in_features": G, "out_features": 10}}]


def build(prefix="v6", layout_fn=None, filters_fn=None):
    rows, feats = (layout_fn or layout)()
    F1, F2 = len(rows), len(feats)
    flt = (filters_fn or v5b.filters)()
    base = sorted({f for f, _ in rows}, key=lambda f: list(flt).index(f))
    B = len(base)
    sd = {"1.weight": np.stack([flt[f][0] for f in base]), "1.bias": np.array([flt[f][1] for f in base])}
    copy = np.zeros((F1, B, 1, 1))
    for r, (f, _) in enumerate(rows):
        copy[r, base.index(f)] = 1.0
    sd["4.weight"], sd["4.bias"] = copy, np.zeros(F1)
    off = np.array([flt[f][2] * w.sum() if len(flt[f]) > 2 else 0.0 for f, w in rows])
    sd["5.weight"], sd["5.bias"] = np.stack([w for _, w in rows])[:, None], -off
    w7 = np.zeros((F2, F1, 1, 1))
    for i, (_, coefs) in enumerate(feats):
        for r, v in coefs.items():
            w7[i, r] = v
    sd["7.weight"], sd["7.bias"] = w7, np.zeros(F2)
    for k, shape in {"9": (F2 * K, 1), "11": (F2, K), "12": (F2 * K2, 1), "14": (2 * F2, K2)}.items():
        sd[f"{k}.weight"], sd[f"{k}.bias"] = np.zeros(shape + (1, 1)), np.zeros(shape[0])
    sd["16.weight"], sd["16.bias"] = np.zeros((10, 2 * F2)), np.zeros(10)
    v2.save(f"{prefix}_s1", arch_lda(B, F1, F2), sd, dict(features=[n for n, _ in feats], F1=F1, F2=F2, B=B,
                                                   filters=base))


def knots(acts_file, prefix="v6"):
    arch, sd, meta = v2.load_net(f"{prefix}_s1")
    f = np.load(acts_file)["acts"]
    F2 = f.shape[1]
    tq = np.arange(K2) / K2
    sq = tq + np.append(tq[1:], 1.0)
    aq = np.concatenate([[sq[0]], np.diff(sq)])
    w9, b9, w11, b11 = np.zeros((F2 * K, 1, 1, 1)), np.zeros(F2 * K), np.zeros((F2, K, 1, 1)), np.zeros(F2)
    w12, b12, w14, b14 = np.zeros((F2 * K2, 1, 1, 1)), np.zeros(F2 * K2), np.zeros((2 * F2, K2, 1, 1)), np.zeros(2 * F2)
    info = {}
    for i in range(F2):
        pos = f[:, i][f[:, i] > 0]
        lo = np.quantile(pos, 0.005) if len(pos) else 1e-3
        hi = max(np.quantile(f[:, i], 0.995), lo * 4)
        t = np.geomspace(lo, hi, K)
        v = np.log(t)
        s = np.diff(v) / np.diff(t)
        w9[i * K:(i + 1) * K, 0, 0, 0], b9[i * K:(i + 1) * K] = 1.0, -t
        w11[i, :, 0, 0], b11[i] = np.concatenate([[s[0]], np.diff(s), [-s[-1]]]), v[0]
        a_, b_ = v[0], v[-1]
        w12[i * K2:(i + 1) * K2, 0, 0, 0] = 1 / (b_ - a_)
        b12[i * K2:(i + 1) * K2] = -a_ / (b_ - a_) - tq
        w14[2 * i, 0, 0, 0] = 1.0
        w14[2 * i + 1, :, 0, 0] = aq
        info[meta["features"][i]] = dict(lo=float(lo), hi=float(hi), share_zero=float((f[:, i] <= 0).mean()))
    sd.update({"9.weight": w9, "9.bias": b9, "11.weight": w11, "11.bias": b11,
               "12.weight": w12, "12.bias": b12, "14.weight": w14, "14.bias": b14})
    meta["knots"] = info
    bad = [k for k, v in info.items() if v["share_zero"] > 0.2]
    print(f"{F2} features; zero for >20% of galaxies: {bad or 'none'}")
    v2.save(f"{prefix}_s2", arch, sd, meta)


def class_stats(acts_file, shrink):
    zf = np.load(acts_file)
    U, y = zf["acts"].astype(np.float64), zf["labels"]
    Cn = 10
    prior = np.bincount(y, minlength=Cn) / len(y)
    mu = np.stack([U[y == k].mean(0) for k in range(Cn)])
    covs = np.stack([np.cov(U[y == k].T) for k in range(Cn)])
    res = U - mu[y]
    Sw = res.T @ res / (len(y) - Cn)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw))
    Sw += 1e-6 * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    return U, y, prior, mu, covs, Sw


def lda(acts_file, temp, shrink, out, prefix="v6"):
    arch, sd, meta = v2.load_net(f"{prefix}_s2")
    U, y, prior, mu, _, Sw = class_stats(acts_file, shrink)
    W = mu @ np.linalg.inv(Sw)
    sd["16.weight"] = temp * W
    sd["16.bias"] = temp * (-0.5 * np.einsum("cf,cf->c", W, mu)) + np.log(prior)
    meta["head"] = dict(kind="lda", temp=temp, shrink=shrink, acts=str(acts_file))
    v2.save(out, arch, sd, meta)


def qda(acts_file, temp, alpha, shrink, out, NQ=NQ, hetero=0.0, prefix="v6"):
    """Regularised QDA in NQ directions; every number from class means and covariances.

    hetero = 0: the leading LDA directions (eigenvectors of the between-class scatter in the whitened space).
    hetero > 0: eigenvectors of S_b + hetero * sum_c prior_c (S_c - I)^2 in the whitened space, so directions in
    which the class spreads differ are kept too (needed when NQ > 9 = classes - 1).
    """
    arch, sd, meta = v2.load_net(f"{prefix}_s2")
    U, y, prior, mu, covs, Sw = class_stats(acts_file, shrink)
    ubar = prior @ mu
    Sb = (mu - ubar).T @ np.diag(prior) @ (mu - ubar)
    Lc = np.linalg.cholesky(Sw)
    Li = np.linalg.inv(Lc)
    Mw = Li @ Sb @ Li.T
    ev, E = np.linalg.eigh(Mw)
    E = E[:, ::-1][:, :min(NQ, 9)]                         # the leading LDA directions are always kept
    if NQ > 9:
        I = np.eye(len(Mw))
        Ms = np.zeros_like(Mw)
        for k in range(10):
            D = Li @ covs[k] @ Li.T - I
            Ms += prior[k] * D @ D                         # where class spreads differ from the pooled one
        P = I - E @ E.T                                    # ... looked for only outside the LDA directions
        _, E2 = np.linalg.eigh(P @ Ms @ P)
        E = np.hstack([E, E2[:, ::-1][:, :NQ - 9]])
    V = Li.T @ E                                           # V' Sw V = I
    muz = (mu - ubar) @ V                                  # (C, NQ)
    covz = np.einsum("fi,cfg,gj->cij", V, covs, V)          # class covariances in z
    totz = V.T @ np.cov(U.T) @ V
    quants = [("sq", (i,)) for i in range(NQ)] + [("pair", ij) for ij in itertools.combinations(range(NQ), 2)]
    G = len(quants) + NQ
    w15, b15 = V.T[:, :, None, None], -(V.T @ ubar)
    w16, b16 = np.zeros((G * 2 * KS, NQ, 1, 1)), np.zeros(G * 2 * KS)
    w18 = np.zeros((G, 2 * KS, 1, 1))
    for g, (_, idx) in enumerate(quants):
        cvec = np.zeros(NQ)
        cvec[list(idx)] = 1.0
        sdv = np.sqrt(cvec @ totz @ cvec)
        tmax = np.abs(muz @ cvec).max() + 4 * sdv
        h = tmax / KS
        t = h * np.arange(KS)
        a = np.full(KS, 2 * h)
        a[0] = h                                            # staircase interpolating |s|^2 at 0, h, 2h, ...
        rows = slice(g * 2 * KS, (g + 1) * 2 * KS)
        w16[rows, :, 0, 0] = np.vstack([np.tile(cvec, (KS, 1)), np.tile(-cvec, (KS, 1))])
        b16[rows] = np.concatenate([-t, -t])
        w18[g, :, 0, 0] = np.concatenate([a, a])
    for i in range(NQ):                                     # carry z_i = relu(z_i) - relu(-z_i)
        g = len(quants) + i
        r0 = g * 2 * KS
        w16[r0, i], w16[r0 + 1, i] = 1.0, -1.0
        b16[r0 + 2:(g + 1) * 2 * KS] = -1.0
        w18[g, 0], w18[g, 1] = 1.0, -1.0
    Wh, bh = np.zeros((10, G)), np.zeros(10)
    for c in range(10):
        S = (1 - alpha) * covz[c] + alpha * np.eye(NQ)
        A = np.linalg.inv(S)
        m = muz[c]
        for g, (kind, idx) in enumerate(quants):
            if kind == "sq":
                i = idx[0]
                Wh[c, g] = -0.5 * (A[i, i] - (A[i].sum() - A[i, i]))
            else:
                Wh[c, g] = -0.5 * A[idx[0], idx[1]]
        Wh[c, len(quants):] = A @ m
        bh[c] = -0.5 * m @ A @ m - 0.5 * np.linalg.slogdet(S)[1]
    B, F1, F2 = meta["B"], meta["F1"], meta["F2"]
    sd = {k: v for k, v in sd.items() if int(k.split(".")[0]) <= 14}
    sd.update({"15.weight": w15, "15.bias": b15, "16.weight": w16, "16.bias": b16, "18.weight": w18,
               "18.bias": np.zeros(G), "20.weight": temp * Wh, "20.bias": temp * bh + np.log(prior)})
    meta["head"] = dict(kind="qda", temp=temp, alpha=alpha, shrink=shrink, nq=NQ, hetero=hetero, acts=str(acts_file))
    v2.save(out, arch_qda(B, F1, F2, NQ), sd, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "lda", "qda"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v6_s3")
    ap.add_argument("--nq", type=int, default=NQ)
    ap.add_argument("--hetero", type=float, default=0.0)
    a = ap.parse_args()
    {"build": build, "knots": lambda: knots(a.acts), "lda": lambda: lda(a.acts, a.temp, a.shrink, a.out),
     "qda": lambda: qda(a.acts, a.temp, a.alpha, a.shrink, a.out, a.nq, a.hetero)}[a.stage]()
