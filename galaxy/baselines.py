"""Reference points for the weight-programming run (the editor does not read these until its edit is frozen).

    python galaxy/baselines.py prior      class frequencies of the training split, scored on val and test
    python galaxy/baselines.py cnn        a conventional small CNN trained with gradients (AdamW, flips/rotations),
                                          temperature-scaled on val; results in runs/baseline_cnn/

Scores use the same `score` as the harness, so they compare directly.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NUM_CLASSES, ROOT, load, to_tensor  # noqa: E402
from wsp import device, report, score  # noqa: E402


def prior():
    _, labels, splits = load()
    freq = np.bincount(labels[splits["train"]], minlength=NUM_CLASSES) / len(splits["train"])
    out = {}
    for split in ("val", "test"):
        y = labels[splits[split]]
        s = score(np.tile(np.log(freq), (len(y), 1)), y)
        out[split] = s
        print(f"== class prior on {split}")
        report(s)
    d = ROOT / "runs" / "baseline_prior"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(out, indent=1))


def block(cin, cout, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


def small_cnn():
    w = [24, 48, 96, 144]                                        # ~477k numbers in the state_dict (limit 500k)
    layers = [nn.AvgPool2d(2)]                                   # 256 -> 128
    cin = 3
    for c in w:
        layers += block(cin, c, 2) + block(c, c)
        cin = c
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(cin, NUM_CLASSES)]
    return nn.Sequential(*layers)


def augment(x):
    """Random flip + 90-degree rotation per image (the notebook's augmentation, done on the device)."""
    k = torch.randint(0, 8, (len(x),), device=x.device)
    out = torch.empty_like(x)
    for t in range(8):
        m = k == t
        if m.any():
            v = x[m].flip(-1) if t >= 4 else x[m]
            out[m] = torch.rot90(v, t % 4, dims=(-2, -1))
    return out


@torch.no_grad()
def logits_of(model, images, idx, dev, bs=256):
    model.eval()
    return np.concatenate([model(to_tensor(images, idx[s:s + bs], dev)).float().cpu().numpy()
                           for s in range(0, len(idx), bs)])


def cnn(epochs=30, bs=64, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    images, labels, splits = load()
    tr, va = splits["train"], splits["val"]
    model = small_cnn().to(dev)
    n_params = sum(v.numel() for v in model.state_dict().values())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    d = ROOT / "runs" / "baseline_cnn"
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "train.log", "w")
    best, t0 = (math.inf, None), time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - bs + 1, bs):
            j = np.sort(perm[s:s + bs])
            x = augment(to_tensor(images, j, dev))
            y = torch.from_numpy(labels[j]).to(dev)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        s = score(logits_of(model, images, va, dev), labels[va])
        print(f"epoch {ep + 1:2d}  val CE {s['cross_entropy']:.4f}  acc {s['accuracy']:.4f}  "
              f"{time.time() - t0:.0f}s", file=log, flush=True)
        if s["cross_entropy"] < best[0]:
            best = (s["cross_entropy"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    lv = torch.from_numpy(logits_of(model, images, va, dev)).double()
    yv = torch.from_numpy(labels[va])
    logt = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # temperature scaling on val (a normal step)
    opt_t = torch.optim.LBFGS([logt], max_iter=100)

    def closure():
        opt_t.zero_grad()
        loss = F.cross_entropy(lv / logt.exp(), yv)
        loss.backward()
        return loss
    opt_t.step(closure)
    T = float(logt.exp())
    out = {"n_params": n_params, "epochs": epochs, "temperature": T, "seconds": round(time.time() - t0)}
    for split in ("val", "test"):
        idx = splits[split]
        out[split] = score(logits_of(model, images, idx, dev) / T, labels[idx])
    (d / "scores.json").write_text(json.dumps(out, indent=1))
    torch.save(model.state_dict(), d / "model.pt")
    print("baseline_cnn done", file=log, flush=True)
    print("done")


if __name__ == "__main__":
    {"prior": prior, "cnn": cnn}[sys.argv[1]]()
