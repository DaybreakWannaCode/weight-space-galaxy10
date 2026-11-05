"""Programmed network v2, written by the editor (Claude). Builds weights with numpy; never runs the network.

Changes from v1, aimed at v1's val confusions (spiral subtypes, disturbed vs loose spiral, cigar vs in-between):
  - the feature Linear reads the 64x64 maps directly (no second pooling), for finer moments and spiral templates;
  - log-spiral templates cos(2 theta - 2 ln r / tan(psi) - phase) at 6 pitch angles (90 = straight bar-like m=2),
    both windings, on both band-pass maps: tight vs loose arms, and straight m=2 for bars;
  - second moments in 5 window sizes (galaxy sizes vary a lot at the fixed pixel scale), and the alignment of inner
    and outer moment vectors: |d_a + c d_b| vs |d_a - c d_b| (a bar twisted against the disk);
  - m=1 lopsidedness of the light and of the band-pass structure; asymmetry of the band-pass map (spiral arms are
    symmetric under 180 degrees, disturbed structure is not);
  - head: LDA on [u, u^2] where u is each log feature rescaled to [0, 1], so class boundaries can be quadratic.

    0  AvgPool2d(4)   1 Conv2d(3->5, 9x9)   2 ReLU   3 Flatten
    4  Linear(20480 -> F1)   5 ReLU   6 Linear(F1 -> F2)   7 ReLU
    8  Linear(F2 -> F2*K)    9 ReLU   10 Linear(F2*K -> F2)              piecewise-linear log, clamped
    11 Linear(F2 -> F2*K2)  12 ReLU   13 Linear(F2*K2 -> 2*F2)           u in [0,1] and piecewise-linear u^2
    14 Linear(2*F2 -> 10)                                                 LDA head (closed form)

    python galaxy/editor/v2.py build                        -> nets/v2_s1 (conv from v1_s2; placeholders after)
    python galaxy/editor/v2.py knots <acts file, layer 7>   -> nets/v2_s2 (log knots; the u, u^2 layers)
    python galaxy/editor/v2.py head  <acts file, layer 13> [--quad 0|1] [--temp T] [--shrink S] --out NAME
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

GAL = Path(__file__).resolve().parent.parent
NETS = GAL / "nets"
K, K2 = 14, 8
N = 64
c = np.arange(N) + 0.5 - N / 2
Y, X = np.meshgrid(c, c, indexing="ij")
R = np.hypot(X, Y)
TH = np.arctan2(Y, X)
CH = ["L", "R", "B", "HPf", "HPc"]
N_DIR = 8
RMAX = 27.0                                   # stay 5 px clear of the 64 px border (conv padding)


def disk(r):
    return np.clip(r + 0.5 - R, 0, 1)


def ramp(r, a, b):
    return np.clip((r - a) / (b - a), 0, 1)


def gwin(s):
    return np.exp(-R ** 2 / (2 * s ** 2)) * disk(RMAX)


def moment_rows(w, phi):
    return w * ((X ** 2 - Y ** 2) * np.cos(phi) + 2 * X * Y * np.sin(phi))


def block_pairs(block, rmax):
    """180-degree pairs of (block x block) cells within rmax (in 64 px units): list of (mask_a, mask_b)."""
    n = N // block
    cc = (np.arange(n) + 0.5 - n / 2) * block
    out = []
    for i in range(n // 2):
        for j in range(n):
            if np.hypot(cc[i], cc[j]) > rmax:
                continue
            a = np.zeros((N, N))
            b = np.zeros((N, N))
            a[i * block:(i + 1) * block, j * block:(j + 1) * block] = 1
            ii, jj = n - 1 - i, n - 1 - j
            b[ii * block:(ii + 1) * block, jj * block:(jj + 1) * block] = 1
            out.append((a, b))
    return out


def layout():
    rows, feats = [], []

    def row(name, wd):
        rows.append((name, wd))
        return len(rows) - 1

    def feat(name, coefs):
        feats.append((name, coefs))

    for r in [3, 5, 7, 10, 14, 20, 27]:
        feat(f"L_disk{r}", {row(f"L_disk{r}", {"L": disk(r)}): 1.0})
    for ch in ("R", "B"):
        for r in [5, 10, 20]:
            feat(f"{ch}_disk{r}", {row(f"{ch}_disk{r}", {ch: disk(r)}): 1.0})
    for ch in ("HPf", "HPc"):
        for r in [8, 16, 27]:
            feat(f"{ch}_disk{r}", {row(f"{ch}_disk{r}", {ch: disk(r)}): 1.0})
    phis = 2 * np.pi * np.arange(N_DIR) / N_DIR
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
    wsp = ramp(R, 2, 4) * (1 - ramp(R, 18, 26))           # annulus for spiral templates
    lnr = np.log(np.maximum(R, 0.5))
    for ch in ("HPc", "HPf"):
        for psi in [12, 18, 25, 35, 50, 90]:
            coefs = {}
            for chir in ([+1, -1] if psi != 90 else [+1]):
                wind = 0.0 if psi == 90 else chir * 2 * lnr / np.tan(np.radians(psi))
                for k, p in enumerate(phis):
                    coefs[row(f"spiral_{ch}_{psi}_{chir}_{k}", {ch: wsp * np.cos(2 * TH - wind - p)})] = 1 / N_DIR
            feat(f"spiral_{ch}_psi{psi}", coefs)
    for ch, w, name in [("L", gwin(8), "m1_L_s8"), ("L", gwin(18), "m1_L_s18"), ("HPc", wsp, "m1_HPc")]:
        feat(name, {row(f"{name}_{k}", {ch: w * (X * np.cos(p) + Y * np.sin(p))}): 1 / N_DIR
                    for k, p in enumerate(phis)})
    for ch, block, rmax, name in [("L", 2, 10, "asym_L_inner"), ("L", 4, RMAX, "asym_L"), ("HPc", 4, RMAX, "asym_HPc")]:
        coefs = {}
        for q, (a, b) in enumerate(block_pairs(block, rmax)):
            coefs[row(f"{name}_{q}+", {ch: a - b})] = 1.0
            coefs[row(f"{name}_{q}-", {ch: b - a})] = 1.0
        feat(name, coefs)
    return rows, feats


def arch_for(F1, F2):
    L = lambda i, o: {"type": "Linear", "args": {"in_features": i, "out_features": o}}  # noqa: E731
    return [
        {"type": "AvgPool2d", "args": {"kernel_size": 4}},
        {"type": "Conv2d", "args": {"in_channels": 3, "out_channels": len(CH), "kernel_size": 9, "padding": 4}},
        {"type": "ReLU"}, {"type": "Flatten"},
        L(len(CH) * N * N, F1), {"type": "ReLU"}, L(F1, F2), {"type": "ReLU"},
        L(F2, F2 * K), {"type": "ReLU"}, L(F2 * K, F2),
        L(F2, F2 * K2), {"type": "ReLU"}, L(F2 * K2, 2 * F2),
        L(2 * F2, 10),
    ]


def save(name, arch, sd, meta):
    d = NETS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "arch.json").write_text(json.dumps(arch, indent=1))
    torch.save({k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in sd.items()}, d / "weights.pt")
    (d / "editor_meta.json").write_text(json.dumps(meta, indent=1))
    n = sum(v.size for v in sd.values())
    nz = sum(int((v != 0).sum()) for v in sd.values())
    print(f"wrote {d}  ({n:,} parameters, {nz:,} nonzero)")


def load_net(name):
    d = NETS / name
    sd = {k: v.numpy().astype(np.float64) for k, v in torch.load(d / "weights.pt", weights_only=True).items()}
    return json.loads((d / "arch.json").read_text()), sd, json.loads((d / "editor_meta.json").read_text())


def build(layout_fn=None, prefix="v2"):
    rows, feats = (layout_fn or layout)()
    F1, F2 = len(rows), len(feats)
    _, sd1, meta1 = load_net("v1_s2")                    # same conv; its biases were set from pass 1
    sd = {"1.weight": sd1["1.weight"], "1.bias": sd1["1.bias"]}
    w4 = np.zeros((F1, len(CH), N, N), np.float32)
    for r, (_, wd) in enumerate(rows):
        for ch, w in wd.items():
            w4[r, CH.index(ch)] = w
    sd["4.weight"], sd["4.bias"] = w4.reshape(F1, -1), np.zeros(F1)
    w6 = np.zeros((F2, F1))
    for f, (_, coefs) in enumerate(feats):
        for r, cf in coefs.items():
            w6[f, r] = cf
    sd["6.weight"], sd["6.bias"] = w6, np.zeros(F2)
    for k, shape in {"8": (F2 * K, F2), "10": (F2, F2 * K), "11": (F2 * K2, F2), "13": (2 * F2, F2 * K2),
                     "14": (10, 2 * F2)}.items():
        sd[f"{k}.weight"], sd[f"{k}.bias"] = np.zeros(shape), np.zeros(shape[0])
    meta = dict(features=[n for n, _ in feats], F1=F1, F2=F2, conv_from="v1_s2", conv_stats=meta1.get("stage2"))
    save(f"{prefix}_s1", arch_for(F1, F2), sd, meta)


def knots(acts_file, prefix="v2"):
    """Log knots (geometric, 0.5%-99.5% quantiles), then u = (log f - a) / (b - a) in [0,1] and its square."""
    arch, sd, meta = load_net(f"{prefix}_s1")
    f = np.load(acts_file)["acts"]
    F2 = f.shape[1]
    w8, b8 = np.zeros((F2 * K, F2)), np.zeros(F2 * K)
    w10, b10 = np.zeros((F2, F2 * K)), np.zeros(F2)
    w11, b11 = np.zeros((F2 * K2, F2)), np.zeros(F2 * K2)
    w13, b13 = np.zeros((2 * F2, F2 * K2)), np.zeros(2 * F2)
    tq = np.arange(K2) / K2
    sq = tq + np.append(tq[1:], 1.0)                     # slopes of u^2 between knots
    aq = np.concatenate([[sq[0]], np.diff(sq)])
    info = {}
    for i in range(F2):
        pos = f[:, i][f[:, i] > 0]
        lo = np.quantile(pos, 0.005) if len(pos) else 1e-3
        hi = max(np.quantile(f[:, i], 0.995), lo * 4)
        t = np.geomspace(lo, hi, K)
        v = np.log(t)
        s = np.diff(v) / np.diff(t)
        w8[i * K:(i + 1) * K, i] = 1.0
        b8[i * K:(i + 1) * K] = -t
        w10[i, i * K:(i + 1) * K] = np.concatenate([[s[0]], np.diff(s), [-s[-1]]])
        b10[i] = v[0]
        a_, b_ = v[0], v[-1]                             # the log output is clamped to [a_, b_]
        w11[i * K2:(i + 1) * K2, i] = 1 / (b_ - a_)
        b11[i * K2:(i + 1) * K2] = -a_ / (b_ - a_) - tq
        w13[i, i * K2] = 1.0                             # u itself (the knot at 0)
        w13[F2 + i, i * K2:(i + 1) * K2] = aq            # piecewise-linear u^2
        info[meta["features"][i]] = dict(lo=float(lo), hi=float(hi), share_zero=float((f[:, i] <= 0).mean()))
    sd.update({"8.weight": w8, "8.bias": b8, "10.weight": w10, "10.bias": b10,
               "11.weight": w11, "11.bias": b11, "13.weight": w13, "13.bias": b13})
    meta["knots"] = info
    bad = [k for k, v in info.items() if v["share_zero"] > 0.2]
    print(f"{F2} features; zero for >20% of galaxies: {bad or 'none'}")
    save(f"{prefix}_s2", arch, sd, meta)


def head(acts_file, quad, temp, shrink, out, prefix="v2"):
    arch, sd, meta = load_net(f"{prefix}_s2")
    zf = np.load(acts_file)
    z, y = zf["acts"].astype(np.float64), zf["labels"]
    F2 = z.shape[1] // 2
    if not quad:
        z = z[:, :F2]
    C = 10
    prior = np.bincount(y, minlength=C) / len(y)
    mu = np.stack([z[y == k].mean(0) for k in range(C)])
    res = z - mu[y]
    S = res.T @ res / (len(y) - C)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    S += 1e-6 * np.trace(S) / len(S) * np.eye(len(S))
    W = mu @ np.linalg.inv(S)
    b = -0.5 * np.einsum("cf,cf->c", W, mu)
    w14 = np.zeros((C, 2 * F2))
    w14[:, :z.shape[1]] = temp * W
    sd["14.weight"], sd["14.bias"] = w14, temp * b + np.log(prior)
    meta["head"] = dict(quad=bool(quad), temp=temp, shrink=shrink, acts=str(acts_file))
    save(out, arch, sd, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "knots", "head"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--quad", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v2_s3")
    a = ap.parse_args()
    {"build": lambda: build(), "knots": lambda: knots(a.acts),
     "head": lambda: head(a.acts, a.quad, a.temp, a.shrink, a.out)}[a.stage]()
