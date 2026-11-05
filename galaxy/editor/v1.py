"""Programmed network v1, written by the editor (Claude). Builds weights with numpy; never runs the network.

The network computes classic galaxy-morphology measurements and classifies them with a closed-form LDA head:

    0  AvgPool2d(4)            256 -> 64 px
    1  Conv2d(3 -> 5, 9x9)     L-b, R-b, B-b (sky-subtracted light and colour), fine and coarse band-pass of L
    2  ReLU                    (biases set the sky level / noise threshold, so only galaxy signal survives)
    3  AvgPool2d(2)            64 -> 32 px
    4  Flatten
    5  Linear(5120 -> F1)      aperture sums, windowed second-moment projections, 180-degree pixel-pair differences
    6  ReLU                    (|x| = relu(x) + relu(-x); the positive part of 8 moment projections sums to |d|)
    7  Linear(F1 -> F2)        F2 positive features: fluxes, band-pass energies, |d|, sizes, asymmetry
    8  ReLU
    9  Linear(F2 -> F2*K)      f - t_k at K knots per feature
    10 ReLU
    11 Linear(F2*K -> F2)      piecewise-linear log(f), clamped outside the knots, so ratios become differences
    12 Linear(F2 -> 10)        LDA: class means and a pooled covariance of the log features (closed form)

    python galaxy/editor/v1.py stage1                          -> nets/v1_s1 (thresholds 0, placeholder head)
    python galaxy/editor/v1.py stage2 <acts file, layer 1>     -> nets/v1_s2 (sky levels / noise thresholds)
    python galaxy/editor/v1.py stage3 <acts file, layer 8>     -> nets/v1_s3 (log knots from feature quantiles)
    python galaxy/editor/v1.py stage4 <acts file, layer 11> [--temp T] [--shrink S]  -> nets/v1_s4 (LDA head)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

GAL = Path(__file__).resolve().parent.parent
NETS = GAL / "nets"
K = 14                                   # knots per feature in the log approximation

# ---------------------------------------------------------------- geometry
N64, N32 = 64, 32
c64 = np.arange(N64) + 0.5 - N64 / 2     # pixel-centre coordinates (units of 4 original px)
c32 = np.arange(N32) + 0.5 - N32 / 2     # (units of 8 original px)
Y32, X32 = np.meshgrid(c32, c32, indexing="ij")
R32 = np.hypot(X32, Y32)

CH = ["L", "R", "B", "HPf", "HPc"]       # conv output channels
L_RADII = [1.5, 2.5, 3.5, 5, 7, 10, 13.5]
COLOR_RADII = [2.5, 5, 10]
HP_RADII = [4, 8, 13.5]
WINDOWS = [2.5, 6.0]                     # Gaussian sigma of the second-moment windows (32-px units)
ASYM_RADII = [5, 13.5]
N_DIR = 8


def disk(r):
    """Anti-aliased disk of radius r on the 32x32 grid."""
    return np.clip(r + 0.5 - R32, 0, 1)


def gauss(sigma, size=9):
    a = np.arange(size) - size // 2
    g = np.exp(-(a[:, None] ** 2 + a[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


ARCH = None   # filled by layout()


def layout():
    """Rows of layer 5 (F1) and layer 7 (F2), as lists of (name, weights) / (name, {row: coef})."""
    rows5, feats = [], []          # rows5: (name, channel weights dict {ch: 32x32 array}); feats: (name, {row: coef})

    def add5(name, wdict):
        rows5.append((name, wdict))
        return len(rows5) - 1

    for r in L_RADII:
        feats.append((f"L_disk{r}", {add5(f"L_disk{r}", {"L": disk(r)}): 1.0}))
    for ch in ("R", "B"):
        for r in COLOR_RADII:
            feats.append((f"{ch}_disk{r}", {add5(f"{ch}_disk{r}", {ch: disk(r)}): 1.0}))
    for ch in ("HPf", "HPc"):
        for r in HP_RADII:
            feats.append((f"{ch}_disk{r}", {add5(f"{ch}_disk{r}", {ch: disk(r)}): 1.0}))
    for s in WINDOWS:
        w = np.exp(-R32 ** 2 / (2 * s ** 2)) * disk(13.5)
        proj = {}
        for k in range(N_DIR):
            phi = 2 * np.pi * k / N_DIR
            # projection of d = (Mxx - Myy, 2 Mxy) on direction phi; the positive parts over 8 directions sum to ~|d|
            proj[add5(f"mom{s}_dir{k}", {"L": w * ((X32 ** 2 - Y32 ** 2) * np.cos(phi) + 2 * X32 * Y32 * np.sin(phi))})] = 1.0 / N_DIR
        feats.append((f"elong_win{s}", proj))
        feats.append((f"size_win{s}", {add5(f"size_win{s}", {"L": w * R32 ** 2}): 1.0}))
        feats.append((f"flux_win{s}", {add5(f"flux_win{s}", {"L": w}): 1.0}))
    for ra in ASYM_RADII:
        pairs = {}
        for i in range(N32 // 2):
            for j in range(N32):
                if R32[i, j] > ra:
                    continue
                a = np.zeros((N32, N32))
                a[i, j], a[N32 - 1 - i, N32 - 1 - j] = 1.0, -1.0
                pairs[add5(f"asym{ra}_{i}_{j}+", {"L": a})] = 1.0
                pairs[add5(f"asym{ra}_{i}_{j}-", {"L": -a})] = 1.0
        feats.append((f"asym{ra}", pairs))
    return rows5, feats


def arch_for(F1, F2):
    return [
        {"type": "AvgPool2d", "args": {"kernel_size": 4}},
        {"type": "Conv2d", "args": {"in_channels": 3, "out_channels": len(CH), "kernel_size": 9, "padding": 4}},
        {"type": "ReLU"},
        {"type": "AvgPool2d", "args": {"kernel_size": 2}},
        {"type": "Flatten"},
        {"type": "Linear", "args": {"in_features": len(CH) * N32 * N32, "out_features": F1}},
        {"type": "ReLU"},
        {"type": "Linear", "args": {"in_features": F1, "out_features": F2}},
        {"type": "ReLU"},
        {"type": "Linear", "args": {"in_features": F2, "out_features": F2 * K}},
        {"type": "ReLU"},
        {"type": "Linear", "args": {"in_features": F2 * K, "out_features": F2}},
        {"type": "Linear", "args": {"in_features": F2, "out_features": 10}},
    ]


def save(name, arch, sd, meta):
    d = NETS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "arch.json").write_text(json.dumps(arch, indent=1))
    torch.save({k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in sd.items()}, d / "weights.pt")
    (d / "editor_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {d}  ({sum(v.size for v in sd.values()):,} parameters)")


def load_net(name):
    d = NETS / name
    sd = {k: v.numpy().astype(np.float64) for k, v in torch.load(d / "weights.pt", weights_only=True).items()}
    return json.loads((d / "arch.json").read_text()), sd, json.loads((d / "editor_meta.json").read_text())


def stage1():
    rows5, feats = layout()
    F1, F2 = len(rows5), len(feats)
    sd = {}
    # conv: luminance, colour channels and band-pass filters (biases = 0 for now)
    w1 = np.zeros((len(CH), 3, 9, 9))
    w1[0, :, 4, 4] = 1 / 3
    w1[1, 0, 4, 4] = 1.0                       # channel 0 of the composite (reddest filter, z)
    w1[2, 2, 4, 4] = 1.0                       # channel 2 (bluest filter, g)
    delta = np.zeros((9, 9))
    delta[4, 4] = 1.0
    hpf, hpc = delta - gauss(1.0), gauss(1.0) - gauss(2.0)
    hpf -= hpf.mean()
    hpc -= hpc.mean()                          # exact zero response to a flat sky
    w1[3] = hpf[None] / 3
    w1[4] = hpc[None] / 3
    sd["1.weight"], sd["1.bias"] = w1, np.zeros(len(CH))
    w5 = np.zeros((F1, len(CH), N32, N32))
    for r, (_, wd) in enumerate(rows5):
        for ch, w in wd.items():
            w5[r, CH.index(ch)] = w
    sd["5.weight"], sd["5.bias"] = w5.reshape(F1, -1), np.zeros(F1)
    w7 = np.zeros((F2, F1))
    for f, (_, coefs) in enumerate(feats):
        for r, c in coefs.items():
            w7[f, r] = c
    sd["7.weight"], sd["7.bias"] = w7, np.zeros(F2)
    sd.update(placeholder_log(F2))
    sd["12.weight"], sd["12.bias"] = np.zeros((10, F2)), np.zeros(10)
    meta = dict(features=[n for n, _ in feats], F1=F1, F2=F2, conv_channels=CH)
    save("v1_s1", arch_for(F1, F2), sd, meta)


def placeholder_log(F2):
    w9 = np.zeros((F2 * K, F2))
    for i in range(F2):
        w9[i * K:(i + 1) * K, i] = 1.0
    w11 = np.zeros((F2, F2 * K))
    for i in range(F2):
        w11[i, i * K] = 1.0
    return {"9.weight": w9, "9.bias": np.zeros(F2 * K), "11.weight": w11, "11.bias": np.zeros(F2)}


def stage2(acts_file):
    """Sky level of L, R, B and noise thresholds of the band-pass channels, from pre-ReLU maps in the corners."""
    arch, sd, meta = load_net("v1_s1")
    z = np.load(acts_file)
    a = z["acts"].reshape(len(z["acts"]), len(CH), N64, N64)
    Y, X = np.meshgrid(c64, c64, indexing="ij")
    sky = (np.hypot(X, Y) > 29) & (np.abs(X) < 27.5) & (np.abs(Y) < 27.5)   # far from the galaxy, clear of padding
    bias = np.zeros(len(CH))
    stats = {}
    for c, name in enumerate(CH):
        v = a[:, c][:, sky]                                    # (n, pixels)
        med = np.median(v, axis=1)                             # per-image sky level
        mad = np.median(np.abs(v - med[:, None]), axis=1) * 1.4826
        if name in ("L", "R", "B"):
            # subtract a typical sky level plus 2 typical noise sigmas, so most sky pixels are cut by the ReLU
            bias[c] = -(np.median(med) + 2 * np.median(mad))
        else:
            bias[c] = -3 * np.median(mad)                      # keep band-pass structure above 3 sigma of sky noise
        stats[name] = dict(sky_median=float(np.median(med)), sky_p90=float(np.quantile(med, 0.9)),
                           noise_median=float(np.median(mad)), bias=float(bias[c]))
    for k, v in stats.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    sd["1.bias"] = bias
    meta["stage2"] = stats
    save("v1_s2", arch, sd, meta)


def stage3(acts_file):
    """Knots of the piecewise-linear log: geometric between the 0.5% and 99.5% quantile of each feature."""
    arch, sd, meta = load_net("v1_s2")
    f = np.load(acts_file)["acts"]
    F2 = f.shape[1]
    w9, b9 = np.zeros((F2 * K, F2)), np.zeros(F2 * K)
    w11, b11 = np.zeros((F2, F2 * K)), np.zeros(F2)
    knots = {}
    for i in range(F2):
        pos = f[:, i][f[:, i] > 0]
        lo = np.quantile(pos, 0.005) if len(pos) else 1e-3
        hi = max(np.quantile(f[:, i], 0.995), lo * 4)
        t = np.geomspace(lo, hi, K)
        v = np.log(t)
        s = np.diff(v) / np.diff(t)
        a = np.concatenate([[s[0]], np.diff(s), [-s[-1]]])     # flat (clamped) below t0 and above t_{K-1}
        w9[i * K:(i + 1) * K, i] = 1.0
        b9[i * K:(i + 1) * K] = -t
        w11[i, i * K:(i + 1) * K] = a
        b11[i] = v[0]
        knots[meta["features"][i]] = dict(lo=float(lo), hi=float(hi), share_zero=float((f[:, i] <= 0).mean()))
    sd.update({"9.weight": w9, "9.bias": b9, "11.weight": w11, "11.bias": b11})
    meta["stage3"] = knots
    for k, v in knots.items():
        print(f"{k:<16s} lo {v['lo']:10.4g}  hi {v['hi']:10.4g}  zero {v['share_zero']:.3f}")
    save("v1_s3", arch, sd, meta)


def stage4(acts_file, temp, shrink, out):
    """LDA head on the log features: logits = T * (mu_c' S^-1 z - mu_c' S^-1 mu_c / 2) + log prior_c."""
    arch, sd, meta = load_net("v1_s3")
    zf = np.load(acts_file)
    z, y = zf["acts"].astype(np.float64), zf["labels"]
    C = 10
    prior = np.bincount(y, minlength=C) / len(y)
    mu = np.stack([z[y == c].mean(0) for c in range(C)])
    resid = z - mu[y]
    S = resid.T @ resid / (len(y) - C)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    S += 1e-6 * np.trace(S) / len(S) * np.eye(len(S))         # keeps a constant feature from making S singular
    Si = np.linalg.inv(S)
    W = mu @ Si                                                # (C, F2)
    b = -0.5 * np.einsum("cf,cf->c", W, mu)
    sd["12.weight"] = temp * W
    sd["12.bias"] = temp * b + np.log(prior)
    meta["stage4"] = dict(temp=temp, shrink=shrink, prior=prior.tolist(), acts=str(acts_file))
    save(out, arch, sd, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "stage3", "stage4"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v1_s4")
    a = ap.parse_args()
    if a.stage == "stage1":
        stage1()
    elif a.stage == "stage2":
        stage2(a.acts)
    elif a.stage == "stage3":
        stage3(a.acts)
    else:
        stage4(a.acts, a.temp, a.shrink, a.out)
