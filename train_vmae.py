"""Model 2: frozen VideoMAE-Kinetics clip features + linear/MLP head.

Each video has W window features (16 frames, stride 8). Train on a random window, eval = mean of window probs.
test_shuffled uses features of the same windows with frames shuffled in time (does the model use motion?).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from common import CLASSES, FRAMES
from train_feats import load_split, metrics


def make_head(dim, n, kind):
    if kind == "linear":
        return nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n))
    return nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n))


@torch.no_grad()
def evaluate(model, items, feats, key="feat"):
    model.eval()
    probs = torch.stack([model(feats[p][key].float()).softmax(-1).mean(0) for p, _ in items])
    return probs, torch.tensor([y for _, y in items])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--split", default="group", choices=["group", "random"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    feats = torch.load(FRAMES.parent / "feats_videomae.pt")
    sp = load_split(a.split)
    model = make_head(768, len(CLASSES), a.head)
    steps = a.epochs * (len(sp["train"]) // 32 + 1)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    counts = np.bincount([y for _, y in sp["train"]], minlength=len(CLASSES))
    w = np.array([1 / counts[y] for _, y in sp["train"]])
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        order = rng.choice(len(sp["train"]), len(sp["train"]), p=w / w.sum())
        for i in range(0, len(order), 32):
            batch = [sp["train"][j] for j in order[i:i + 32]]
            x = torch.stack([(f := feats[p]["feat"])[rng.integers(len(f))].float() for p, _ in batch])
            loss = Fn.cross_entropy(model(x), torch.tensor([y for _, y in batch]), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = metrics(*evaluate(model, sp["val"], feats))
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"val": metrics(*evaluate(model, sp["val"], feats)), "test": metrics(*evaluate(model, sp["test"], feats)),
           "test_shuffled": metrics(*evaluate(model, sp["test"], feats, "feat_shuf"))}
    print(json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()}))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "head": a.head}, f"{a.out}/model.pt")
        json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
