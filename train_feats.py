"""Model 1: frozen image encoder (precomputed per-frame features) + small head.

--head mean      : MLP on the average of 16 frame features (bag of frames, no order)
--head temporal  : 2-layer transformer over the 16 frames (can use order/motion)
Eval averages logits over all 16-frame windows of a video (stride 8). --shuffle-test permutes frame order at test
time: if accuracy doesn't drop, the model isn't using motion.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from common import CLASSES, FRAMES, sample_clip

T = 16


class Head(nn.Module):
    def __init__(self, dim, n_cls, kind, width=256):
        super().__init__()
        self.kind = kind
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3), nn.Linear(dim, width), nn.GELU())
        if kind == "temporal":
            self.pos = nn.Parameter(torch.zeros(1, T + 1, width))
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.2, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.3), nn.Linear(width, n_cls))

    def forward(self, x):  # (B, T, D)
        h = self.proj(x)
        if self.kind == "mean":
            return self.out(h.mean(1))
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos
        return self.out(self.tf(h)[:, 0])


def windows(n):
    if n <= T:
        return [sample_clip(n, T)]
    return [np.arange(s, s + T) for s in range(0, n - T + 1, 8)]


@torch.no_grad()
def evaluate(model, items, feats, shuffle=False, rng=None):
    model.eval()
    probs = []
    for path, _ in items:
        f = feats[path].float()
        ws = [w if not shuffle else rng.permutation(w) for w in windows(len(f))]
        logits = model(torch.stack([f[w] for w in ws]))
        probs.append(logits.softmax(-1).mean(0))
    probs = torch.stack(probs)
    return probs, torch.tensor([y for _, y in items])


def metrics(probs, y):
    pred = probs.argmax(1)
    f1 = []
    for c in range(len(CLASSES)):
        tp = ((pred == c) & (y == c)).sum().item()
        p, r = (pred == c).sum().item(), (y == c).sum().item()
        if r:
            f1.append(2 * tp / (p + r) if p + r else 0)
    return {"acc": (pred == y).float().mean().item(), "macro_f1": float(np.mean(f1))}


def load_split(name):
    rows = list(csv.DictReader(open(FRAMES.parent / f"split_{name}.csv")))
    return {s: [(r["path"], CLASSES.index(r["label"])) for r in rows if r["split"] == s] for s in ("train", "val", "test")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="dinov2s")
    ap.add_argument("--head", default="temporal", choices=["mean", "temporal"])
    ap.add_argument("--split", default="group", choices=["group", "random"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    feats = torch.load(FRAMES.parent / f"feats_{a.feats}.pt")
    sp = load_split(a.split)
    dim = next(iter(feats.values())).shape[1]
    model = Head(dim, len(CLASSES), a.head)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.epochs * (len(sp["train"]) // 32 + 1))
    # class-balanced sampling weights
    counts = np.bincount([y for _, y in sp["train"]], minlength=len(CLASSES))
    w = np.array([1 / counts[y] for _, y in sp["train"]])
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        order = rng.choice(len(sp["train"]), len(sp["train"]), p=w / w.sum())
        for i in range(0, len(order), 32):
            batch = [sp["train"][j] for j in order[i:i + 32]]
            x = torch.stack([feats[p].float()[sample_clip(len(feats[p]), T, rng)] for p, _ in batch])
            x = x + 0.05 * torch.randn_like(x) * x.std()
            loss = Fn.cross_entropy(model(x), torch.tensor([y for _, y in batch]), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = metrics(*evaluate(model, sp["val"], feats))
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"val": metrics(*evaluate(model, sp["val"], feats)), "test": metrics(*evaluate(model, sp["test"], feats)),
           "test_shuffled": metrics(*evaluate(model, sp["test"], feats, shuffle=True, rng=rng))}
    print(json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()}))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "head": a.head, "feats": a.feats, "dim": dim}, f"{a.out}/model.pt")
        json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
