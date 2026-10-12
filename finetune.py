"""Model 3: fine-tune the top blocks of VideoMAE-Kinetics end to end, with video augmentation.

Frozen bottom --freeze blocks (generic motion features) + trainable top blocks + new 22-class head.
Augment: random temporal window, random resized crop (same for all frames), h-flip, brightness/contrast.
Saves runs/<out>/model.pt (trainable weights only) + results.json (test, test_shuffled).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from common import CLASSES, FRAMES, sample_clip
from features_video import MEAN, STD, load, windows
from train_feats import load_split, metrics

T = 16


def augment(x, rng):
    """x: (B, T, 3, H, W) float in [0,1] on device"""
    B, _, _, H, W = x.shape
    out = []
    for b in range(B):
        s = rng.uniform(0.55, 1.0); ar = rng.uniform(0.8, 1.25)
        h, w = min(H, int(H * s / ar ** 0.5)), min(W, int(W * s * ar ** 0.5))
        y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        c = Fn.interpolate(x[b, :, :, y0:y0 + h, x0:x0 + w], size=(H, W), mode="bilinear", align_corners=False)
        if rng.random() < 0.5:
            c = c.flip(-1)
        c = (c - c.mean()) * rng.uniform(0.7, 1.3) + c.mean() + rng.uniform(-0.1, 0.1)
        out.append(c.clamp(0, 1))
    return torch.stack(out)


def to_input(clips_uint8, dev):
    return torch.from_numpy(np.ascontiguousarray(clips_uint8)).to(dev).permute(0, 1, 4, 2, 3).float() / 255


class Model(nn.Module):
    def __init__(self, vmae, n_cls):
        super().__init__()
        self.vmae, self.norm = vmae.videomae, vmae.fc_norm
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, n_cls))

    def forward(self, x):  # (B, T, 3, H, W) in [0,1]
        h = self.vmae(pixel_values=(x - MEAN.to(x.device)) / STD.to(x.device)).last_hidden_state
        return self.head(self.norm(h.mean(1)))


@torch.no_grad()
def evaluate(model, items, dev, shuffle=False, rng=None, bs=8):
    model.eval()
    probs = []
    for p, _ in items:
        v = np.load(FRAMES / p)
        ws = [rng.permutation(w) if shuffle else w for w in windows(len(v))]
        logits = torch.cat([model(to_input(np.stack([v[w] for w in ws[i:i + bs]]), dev)) for i in range(0, len(ws), bs)])
        probs.append(logits.softmax(-1).mean(0).cpu())
    return torch.stack(probs), torch.tensor([y for _, y in items])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="group", choices=["group", "random"])
    ap.add_argument("--freeze", type=int, default=8, help="freeze patch embed + this many of the 12 blocks")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/finetune")
    a = ap.parse_args()
    dev = "mps"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = Model(load(dev), len(CLASSES)).to(dev)
    for p in model.vmae.embeddings.parameters():
        p.requires_grad = False
    for blk in model.vmae.encoder.layer[:a.freeze]:
        for p in blk.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M params")
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in params if all(p is not q for q in head_params)], "lr": a.lr},
                             {"params": head_params, "lr": a.lr * 20}], weight_decay=0.05)
    sp = load_split(a.split)
    steps = a.epochs * (len(sp["train"]) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr * 20], total_steps=steps, pct_start=0.1)
    counts = np.bincount([y for _, y in sp["train"]], minlength=len(CLASSES))
    w = np.array([1 / counts[y] ** 0.5 for _, y in sp["train"]])  # soften imbalance
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        model.train()
        order = rng.choice(len(sp["train"]), len(sp["train"]) // a.bs * a.bs, p=w / w.sum())
        tot = 0.0
        for i in range(0, len(order), a.bs):
            batch = [sp["train"][j] for j in order[i:i + a.bs]]
            clips = []
            for p, _ in batch:
                v = np.load(FRAMES / p, mmap_mode="r")
                clips.append(v[sample_clip(len(v), T, rng)])
            x = augment(to_input(np.stack(clips), dev), rng)
            loss = Fn.cross_entropy(model(x), torch.tensor([y for _, y in batch], device=dev), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        m = metrics(*evaluate(model, sp["val"], dev))
        print(f"epoch {ep + 1}: loss {tot / (len(order) // a.bs):.3f} val acc {m['acc']:.3f} f1 {m['macro_f1']:.3f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"val": metrics(*evaluate(model, sp["val"], dev)), "test": metrics(*evaluate(model, sp["test"], dev)),
           "test_shuffled": metrics(*evaluate(model, sp["test"], dev, shuffle=True, rng=rng))}
    print(json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()}))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "freeze": a.freeze}, f"{a.out}/model.pt")
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
