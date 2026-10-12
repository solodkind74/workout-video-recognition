"""Frozen VideoMAE (Kinetics-400 fine-tuned) on every 16-frame window (stride 8) -> data/feats_videomae.pt

Stores {path: {"feat": (W, 768), "k400": (W, 400) logits, "feat_shuf": (W, 768) with frames shuffled in time}}.
"""
import csv

import numpy as np
import torch
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification

from common import FRAMES, sample_clip

NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def windows(n, t=16, stride=8):
    return [sample_clip(n, t)] if n <= t else [np.arange(s, s + t) for s in range(0, n - t + 1, stride)]


def load(dev):
    return VideoMAEForVideoClassification.from_pretrained(NAME).to(dev).eval()


@torch.no_grad()
def run(model, clips_uint8, dev, bs=8):
    """(W, 16, H, W, 3) uint8 -> pooled features (W, 768), logits (W, 400)"""
    feats, logits = [], []
    for i in range(0, len(clips_uint8), bs):
        x = torch.from_numpy(np.ascontiguousarray(clips_uint8[i:i + bs])).permute(0, 1, 4, 2, 3).float() / 255
        x = ((x - MEAN) / STD).to(dev)
        h = model.videomae(pixel_values=x).last_hidden_state
        f = model.fc_norm(h.mean(1))
        feats.append(f.cpu()); logits.append(model.classifier(f).cpu())
    return torch.cat(feats), torch.cat(logits)


if __name__ == "__main__":
    dev = "mps"
    m = load(dev)
    rng = np.random.default_rng(0)
    out = {}
    for r in tqdm(list(csv.DictReader(open(FRAMES.parent / "index.csv"))), mininterval=30):
        v = np.load(FRAMES / r["path"])
        ws = windows(len(v))
        f, k = run(m, np.stack([v[w] for w in ws]), dev)
        fs, _ = run(m, np.stack([v[rng.permutation(w)] for w in ws]), dev)
        out[r["path"]] = {"feat": f.half(), "k400": k.half(), "feat_shuf": fs.half()}
    torch.save(out, FRAMES.parent / "feats_videomae.pt")
    print(len(out), "videos")
