"""What does Kinetics-400 call each exercise? + confusions of the frozen VideoMAE linear head on the group test split."""
import csv
from collections import Counter
import torch
from transformers import AutoConfig
from common import CLASSES, FRAMES
from features_video import NAME
from train_vmae import evaluate, make_head
from train_feats import load_split
k400 = AutoConfig.from_pretrained(NAME).id2label
F = torch.load(FRAMES.parent / "feats_videomae.pt")
rows = list(csv.DictReader(open(FRAMES.parent / "index.csv")))
print("== top Kinetics label per exercise (share of videos)")
for c in CLASSES:
    top = Counter(k400[int(F[r["path"]]["k400"].float().mean(0).argmax())] for r in rows if r["label"] == c)
    n = sum(top.values())
    print(f"{c:22s}", ", ".join(f"{k} {v / n:.0%}" for k, v in top.most_common(2)))
ck = torch.load("runs/vmae_linear_group/model.pt"); h = make_head(768, len(CLASSES), ck["head"]); h.load_state_dict(ck["state"])
p, y = evaluate(h, load_split("group")["test"], F)
print("== test confusions (true -> predicted)")
for (t, q), n in Counter((CLASSES[a], CLASSES[b]) for a, b in zip(y.tolist(), p.argmax(1).tolist()) if a != b).most_common():
    print(f"  {t} -> {q} x{n}")
