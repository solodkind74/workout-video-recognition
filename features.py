"""Frozen image encoder on every stored frame -> data/feats_<name>.pt  {path: (T, D) float16}"""
import argparse
import csv

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from common import FRAMES

MODELS = {"dinov2s": "facebook/dinov2-small", "dinov2b": "facebook/dinov2-base"}
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(name, dev):
    return AutoModel.from_pretrained(MODELS[name]).to(dev).eval()


@torch.no_grad()
def embed(model, frames_uint8, dev, bs=64):
    """(T, H, W, 3) uint8 -> (T, 2D): [CLS, mean patch]"""
    out = []
    for i in range(0, len(frames_uint8), bs):
        x = torch.from_numpy(np.ascontiguousarray(frames_uint8[i:i + bs])).permute(0, 3, 1, 2).float() / 255
        x = ((x - MEAN) / STD).to(dev)
        h = model(pixel_values=x).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).half().cpu())
    return torch.cat(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2s", choices=MODELS)
    a = ap.parse_args()
    dev = "mps"
    m = load(a.model, dev)
    rows = list(csv.DictReader(open(FRAMES.parent / "index.csv")))
    feats = {r["path"]: embed(m, np.load(FRAMES / r["path"]), dev) for r in tqdm(rows, mininterval=20)}
    torch.save(feats, FRAMES.parent / f"feats_{a.model}.pt")
    print(len(feats), "videos, dim", next(iter(feats.values())).shape[1])
