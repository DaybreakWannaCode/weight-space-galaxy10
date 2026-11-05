"""Programmed network v4, written by the editor (Claude): v3 plus a tightness measure from edge orientation.

Tight arms are nearly circular, so the light changes along the radius; loose arms run outward, so it also changes
around the circle. Three derivative-of-Gaussian channels (directions 0, 60, 120 degrees; positive part above the sky
noise) feed rows that weight each pixel's channel k by cos^2(phi_k - theta) (radial) or sin^2 (tangential), theta
being the pixel's position angle. Radial and tangential edge energy in 3 rings; the LDA head sees their log ratio.
To stay under 30M parameters the c=3 ring-turn features of v3 are dropped (no separation on train).

    python galaxy/editor/v4.py build                   -> nets/v4_s0 (derivative thresholds 0)
    python galaxy/editor/v4.py thresholds <acts layer 1>  -> nets/v4_s1
    python galaxy/editor/v4.py knots <acts layer 7>    -> nets/v4_s2
    python galaxy/editor/v4.py head <acts layer 13> [--quad] [--temp] [--shrink] --out NAME
"""
import argparse
import json

import numpy as np

import v2
import v3
from v2 import N, R, TH, X, Y

CH = v2.CH + ["D0", "D60", "D120"]
DIRS = [0, 60, 120]
ZONES = [(3, 8), (8, 15), (15, 24)]


def layout():
    rows, feats = v3.layout()
    keep = [(n, c) for n, c in feats if not (n.startswith("turn") and "_c3_" in n)]
    used = sorted({r for _, c in keep for r in c})
    remap = {r: i for i, r in enumerate(used)}
    rows = [rows[r] for r in used]
    feats = [(n, {remap[r]: v for r, v in c.items()}) for n, c in keep]
    for a, b in ZONES:
        w = np.clip(np.minimum(R - a + 0.5, b + 0.5 - R), 0, 1)
        for kind, f in (("rad", np.cos), ("tan", np.sin)):
            wd = {f"D{d}": w * f(np.radians(d) - TH) ** 2 for d in DIRS}
            rows.append((f"edge_{kind}_{a}_{b}", wd))
            feats.append((f"edge_{kind}_{a}_{b}", {len(rows) - 1: 1.0}))
    return rows, feats


def arch_for(F1, F2):
    arch = v2.arch_for(F1, F2)
    arch[1]["args"]["out_channels"] = len(CH)
    arch[4]["args"]["in_features"] = len(CH) * N * N
    return arch


def build():
    rows, feats = layout()
    F1, F2 = len(rows), len(feats)
    _, sd1, meta1 = v2.load_net("v1_s2")
    w1 = np.zeros((len(CH), 3, 9, 9))
    w1[:5] = sd1["1.weight"]
    a = np.arange(9) - 4
    dy, dx = np.meshgrid(a, a, indexing="ij")
    g = np.exp(-(dx ** 2 + dy ** 2) / 2.0)
    g /= g.sum()
    for k, d in enumerate(DIRS):
        ker = (dx * np.cos(np.radians(d)) + dy * np.sin(np.radians(d))) * g     # sigma = 1: derivative along d
        w1[5 + k] = ker[None] / 3
    sd = {"1.weight": w1, "1.bias": np.concatenate([sd1["1.bias"], np.zeros(len(DIRS))])}
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
    K, K2 = v2.K, v2.K2
    for k, shape in {"8": (F2 * K, F2), "10": (F2, F2 * K), "11": (F2 * K2, F2), "13": (2 * F2, F2 * K2),
                     "14": (10, 2 * F2)}.items():
        sd[f"{k}.weight"], sd[f"{k}.bias"] = np.zeros(shape), np.zeros(shape[0])
    meta = dict(features=[n for n, _ in feats], F1=F1, F2=F2, channels=CH, conv_from="v1_s2")
    v2.save("v4_s0", arch_for(F1, F2), sd, meta)


def thresholds(acts_file):
    arch, sd, meta = v2.load_net("v4_s0")
    z = np.load(acts_file)["acts"]
    a = z.reshape(len(z), len(CH), N, N)
    sky = (R > 29) & (np.abs(X) < 27.5) & (np.abs(Y) < 27.5)
    out = {}
    for k in range(5, len(CH)):
        v = a[:, k][:, sky]
        med = np.median(v, axis=1)
        mad = np.median(np.abs(v - med[:, None]), axis=1) * 1.4826
        sd["1.bias"][k] = -2 * np.median(mad)
        out[CH[k]] = dict(noise=float(np.median(mad)), bias=float(sd["1.bias"][k]))
    print(json.dumps(out))
    meta["derivative_thresholds"] = out
    v2.save("v4_s1", arch, sd, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "thresholds", "knots", "head"])
    ap.add_argument("acts", nargs="?")
    ap.add_argument("--quad", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--out", default="v4_s3")
    a = ap.parse_args()
    if a.stage == "build":
        build()
    elif a.stage == "thresholds":
        thresholds(a.acts)
    elif a.stage == "knots":
        v2.knots(a.acts, "v4")
    else:
        v2.head(a.acts, a.quad, a.temp, a.shrink, a.out, "v4")
