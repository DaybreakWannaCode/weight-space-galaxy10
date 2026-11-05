"""Weight-space programming harness for the Galaxy10 challenge: the only way a programmed network is ever run.

    python galaxy/wsp.py check  <net>                          free: architecture, parameter count, layer shapes
    python galaxy/wsp.py acts   <net> --upto K [--n N] [--seed S]   1 pass: activations after layer K, training images
    python galaxy/wsp.py val    <net>                          1 pass: cross-entropy, accuracy, confusion (val split)
    python galaxy/wsp.py final  <net>                          once per run, no pass cost: the test score + submission
    python galaxy/wsp.py status

All commands take --run NAME (default r1); a run's budget is fixed when it is first used (--budget, default 50).
<net> is a folder holding arch.json (a list of {"type": ..., "args": {...}}) and weights.pt (its state_dict).

Rules for the editor (the frontier model writing the weights):
  1. The network is an nn.Sequential of the layer types in ALLOWED, so forward() is fixed and every behaviour has to
     live in the weights. Fewer than 500,000 numbers in the state_dict. It gets the notebook's input: (B, 3, 256,
     256) float in [0, 1], and must output 10 logits; the score is mean cross-entropy (natural log).
  2. No gradients and no fitting: no autograd, backward passes, optimizers, and no estimator that minimises a loss
     (least squares, logistic regression, iterative search over weights against a score). Closed-form statistics of
     data or of returned activations are allowed: counts, means, covariances, quantiles, eigenvectors (this covers
     difference-of-means directions and LDA, built from class means and a covariance).
  3. Training images and labels may be read and inspected directly (numpy, viewing images).
  4. The network runs only through this tool: never in the editor's own code. Each `acts` or `val` call costs one
     pass; failed calls cost nothing. The val split is seen only through `val`'s summary numbers; the test split only
     through `final`, which is run once, on the edit the editor submits.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, MAX_PARAMS, NUM_CLASSES, ROOT, load, to_tensor  # noqa: E402

ALLOWED = {
    "Conv2d": {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"},
    "Linear": {"in_features", "out_features", "bias"},
    "ReLU": set(),
    "AvgPool2d": {"kernel_size", "stride", "padding"},
    "MaxPool2d": {"kernel_size", "stride", "padding", "dilation"},
    "AdaptiveAvgPool2d": {"output_size"},
    "AdaptiveMaxPool2d": {"output_size"},
    "Flatten": set(),
}
MAX_ACT_FLOATS = 150_000_000          # cap on one `acts` result (600 MB of float32)
BATCH = 128


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build(arch):
    """nn.Sequential from an arch list, refusing anything outside ALLOWED."""
    layers = []
    for i, spec in enumerate(arch):
        t, args = spec["type"], spec.get("args", {})
        if t not in ALLOWED:
            raise ValueError(f"layer {i}: type {t!r} is not allowed")
        if set(args) - ALLOWED[t]:
            raise ValueError(f"layer {i}: arguments {sorted(set(args) - ALLOWED[t])} are not allowed for {t}")
        for k, v in args.items():
            ok = isinstance(v, (bool, int)) or (isinstance(v, list) and all(isinstance(e, int) for e in v))
            if not ok:
                raise ValueError(f"layer {i}: argument {k}={v!r} must be an int, bool or list of ints")
        layers.append(getattr(nn, t)(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()}))
    return nn.Sequential(*layers)


def shapes(arch):
    """Output shape after each layer, computed on the meta device (nothing is run)."""
    model = build(arch).to("meta")
    x = torch.empty(1, 3, 256, 256, device="meta")
    out = []
    for layer in model:
        x = layer(x)
        out.append(tuple(x.shape[1:]))
    return out


def load_net(net_dir):
    net_dir = Path(net_dir)
    arch = json.loads((net_dir / "arch.json").read_text())
    model = build(arch)
    sd = torch.load(net_dir / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    sd = model.state_dict()
    n_params = sum(v.numel() for v in sd.values())
    if n_params >= MAX_PARAMS:
        raise ValueError(f"{n_params:,} parameters; the limit is {MAX_PARAMS:,}")
    for k, v in sd.items():
        if not torch.isfinite(v).all():
            raise ValueError(f"{k} has non-finite values")
    h = hashlib.sha256(json.dumps(arch, sort_keys=True).encode())
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().float().numpy().tobytes())
    out_shape = shapes(arch)[-1]
    if out_shape != (NUM_CLASSES,):
        raise ValueError(f"the network outputs {out_shape}, not ({NUM_CLASSES},)")
    return model.eval(), arch, n_params, h.hexdigest()[:16]


class Run:
    def __init__(self, name, budget):
        self.dir = ROOT / "runs" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = dict(budget=budget, used=0, final=None, created=time.strftime("%Y-%m-%d %H:%M:%S"))
            self.save()

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def need_pass(self):
        if self.state["used"] >= self.state["budget"]:
            raise SystemExit(f"budget used up ({self.state['used']}/{self.state['budget']} passes)")

    def spend(self, entry):
        self.state["used"] += 1
        entry = dict(pass_no=self.state["used"], time=time.strftime("%Y-%m-%d %H:%M:%S"), **entry)
        self.save()
        self.log(entry)
        return entry["pass_no"]

    def log(self, entry):
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


@torch.no_grad()
def forward(model, images, idx, upto=None):
    dev = device()
    model = model.to(dev)
    part = model if upto is None else model[:upto + 1]
    outs = []
    for s in range(0, len(idx), BATCH):
        outs.append(part(to_tensor(images, idx[s:s + BATCH], dev)).flatten(1).float().cpu())
    return torch.cat(outs).numpy()


def score(logits, y):
    logits = torch.from_numpy(logits).double()
    logp = torch.log_softmax(logits, 1)
    nll = -logp[torch.arange(len(y)), torch.from_numpy(y)].numpy()
    p = logp.exp().numpy()
    pred = p.argmax(1)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    np.add.at(conf, (y, pred), 1)
    maxp = p.max(1)
    bins = np.minimum((maxp * 10).astype(int), 9)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - maxp[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return dict(
        cross_entropy=float(nll.mean()), accuracy=float((pred == y).mean()), ece=float(ece),
        mean_max_prob=float(maxp.mean()), share_nll_over_5=float((nll > 5).mean()), max_nll=float(nll.max()),
        nll_by_class=[float(nll[y == c].mean()) for c in range(NUM_CLASSES)],
        recall_by_class=[float((pred[y == c] == c).mean()) for c in range(NUM_CLASSES)],
        confusion=conf.tolist(),
    )


def report(s):
    print(f"cross-entropy {s['cross_entropy']:.4f}  accuracy {s['accuracy']:.4f}  ECE {s['ece']:.4f}  "
          f"mean max prob {s['mean_max_prob']:.3f}  NLL>5 {s['share_nll_over_5']:.3%}  max NLL {s['max_nll']:.2f}")
    print(f"{'class':<27s}{'NLL':>7s}{'recall':>8s}   confusion row (predicted 0..9)")
    for c in range(NUM_CLASSES):
        row = " ".join(f"{v:4d}" for v in s["confusion"][c])
        print(f"{c} {CLASS_NAMES[c]:<25s}{s['nll_by_class'][c]:7.3f}{s['recall_by_class'][c]:8.3f}   {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "acts", "val", "final", "status"])
    ap.add_argument("net", nargs="?")
    ap.add_argument("--run", default="r1")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--upto", type=int, help="acts: the layer index whose output is returned")
    ap.add_argument("--n", type=int, help="acts: a random subset of this many training images (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Run(a.run, a.budget)

    if a.cmd == "status":
        print(json.dumps(run.state, indent=1))
        return
    model, arch, n_params, h = load_net(a.net)
    if a.cmd == "check":
        for i, (spec, shp) in enumerate(zip(arch, shapes(arch))):
            print(f"{i:3d} {spec['type']:<18s} -> {shp}")
        print(f"{n_params:,} parameters, hash {h}; passes used {run.state['used']}/{run.state['budget']}")
        return

    images, labels, splits = load()
    t0 = time.time()
    if a.cmd == "acts":
        run.need_pass()
        idx = splits["train"]
        if a.n:
            idx = np.sort(np.random.default_rng(a.seed).choice(idx, a.n, replace=False))
        if a.upto is None or not 0 <= a.upto < len(arch):
            raise SystemExit(f"--upto must be a layer index 0..{len(arch) - 1}")
        width = int(np.prod(shapes(arch)[a.upto]))
        if width * len(idx) > MAX_ACT_FLOATS:
            raise SystemExit(f"{width} x {len(idx)} values is over the {MAX_ACT_FLOATS:,} cap; use --n or an earlier "
                             f"pooling layer")
        acts = forward(model, images, idx, a.upto)
        out = run.dir / "acts" / f"pass{run.state['used'] + 1:03d}_layer{a.upto}.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez(out, acts=acts, labels=labels[idx], indices=idx)
        k = run.spend(dict(cmd="acts", net=str(a.net), hash=h, upto=a.upto, n=len(idx), file=str(out),
                           seconds=round(time.time() - t0, 1)))
        print(f"pass {k}/{run.state['budget']}: activations {acts.shape} after layer {a.upto} -> {out}")
    elif a.cmd == "val":
        run.need_pass()
        idx = splits["val"]
        s = score(forward(model, images, idx), labels[idx])
        k = run.spend(dict(cmd="val", net=str(a.net), hash=h, n_params=n_params, seconds=round(time.time() - t0, 1),
                           **s))
        print(f"pass {k}/{run.state['budget']} (val, {len(idx)} galaxies, {n_params:,} parameters, hash {h})")
        report(s)
    elif a.cmd == "final":
        if run.state["final"]:
            raise SystemExit(f"this run's final evaluation was already done: {run.state['final']}")
        idx = splits["test"]
        s = score(forward(model, images, idx), labels[idx])
        sub = run.dir / "submission"
        sub.mkdir(exist_ok=True)
        torch.save(model.cpu().state_dict(), sub / "submission.pt")
        (sub / "arch.json").write_text(json.dumps(arch, indent=1))
        run.state["final"] = dict(net=str(a.net), hash=h, cross_entropy=s["cross_entropy"], accuracy=s["accuracy"])
        run.save()
        run.log(dict(cmd="final", net=str(a.net), hash=h, n_params=n_params, **s))
        print(f"TEST ({len(idx)} galaxies, {n_params:,} parameters, hash {h})")
        report(s)


if __name__ == "__main__":
    main()
