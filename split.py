"""Group near-duplicate clips (same source video/person/gym) and split by group -> data/split_{group,random}.csv

Many clips are cut from the same YouTube video. Groups = single-linkage clusters of same-class clips whose mean
DINOv2 CLS cosine > --thresh (eyeballed: 0.85 pairs were the same person in the same gym, 0.75 different people).
Linking only within a class stops one generic "gym" look from chaining everything into a single blob.
"""
import argparse
import csv
from collections import Counter

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from common import FRAMES

ap = argparse.ArgumentParser()
ap.add_argument("--thresh", type=float, default=0.8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = list(csv.DictReader(open(FRAMES.parent / "index.csv")))
lab = np.array([r["label"] for r in rows])
F = torch.load(FRAMES.parent / "feats_dinov2s.pt")
E = torch.nn.functional.normalize(torch.stack([F[r["path"]][:, :384].float().mean(0) for r in rows]), dim=1)
D = 1 - (E @ E.T).numpy()
D[lab[:, None] != lab[None]] = 2
np.fill_diagonal(D, 0)
group = fcluster(linkage(squareform(np.clip((D + D.T) / 2, 0, None), checks=False), "single"), 1 - a.thresh, "distance")
sizes = Counter(group)
print(f"{len(sizes)} groups from {len(rows)} videos; largest {sizes.most_common(1)[0][1]}; "
      f"singletons {sum(s == 1 for s in sizes.values())}")

# per class: biggest group -> train, then shuffled groups fill test and val to ~15% each, rest train
rng = np.random.default_rng(a.seed)
split = {}
for cls in sorted(set(lab)):
    gs = sorted({g for g in group[lab == cls]}, key=lambda g: -sizes[g])
    first, rest = gs[0], list(rng.permutation(gs[1:]))
    n, filled = (lab == cls).sum(), {"test": 0, "val": 0}
    split[first] = "train"
    for g in rest:
        s = next((k for k in ("test", "val") if filled[k] == 0 or filled[k] + sizes[g] <= 0.15 * n + 1), "train")
        filled[s] = filled.get(s, 0) + sizes[g]
        split[g] = s
gsplit = [split[g] for g in group]
order = rng.permutation(len(rows))
rsplit = [None] * len(rows)
for k, i in enumerate(order):
    rsplit[i] = "test" if k < 0.15 * len(rows) else "val" if k < 0.3 * len(rows) else "train"

for name, sp in (("group", gsplit), ("random", rsplit)):
    with open(FRAMES.parent / f"split_{name}.csv", "w") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "path", "frames", "group", "split"])
        for i, r in enumerate(rows):
            w.writerow([r["label"], r["path"], r["frames"], group[i], sp[i]])
    print(name, dict(Counter(sp)))
train_groups = {group[i] for i in range(len(rows)) if rsplit[i] == "train"}
test = [i for i in range(len(rows)) if rsplit[i] == "test"]
print(f"random split: {sum(group[i] in train_groups for i in test)}/{len(test)} test videos have a same-source video in train")
