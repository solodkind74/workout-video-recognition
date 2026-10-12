"""Near-duplicate check: nearest-neighbor cosine of mean CLS embeddings; saves pair sheets for eyeballing."""
import csv, sys
import numpy as np, torch, cv2
from common import FRAMES
rows = list(csv.DictReader(open(FRAMES.parent / "index.csv")))
F = torch.load(FRAMES.parent / "feats_dinov2s.pt")
E = torch.stack([F[r["path"]][:, :384].float().mean(0) for r in rows])
E = torch.nn.functional.normalize(E, dim=1)
S = E @ E.T; S.fill_diagonal_(-1)
nn = S.max(1)
v = nn.values.numpy()
for t in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
    same = sum(rows[i]["label"] == rows[j]["label"] for i, j in enumerate(nn.indices.tolist()) if v[i] > t)
    print(f"nn sim > {t}: {(v > t).sum():4d} videos ({same} same class)")
lab = np.array([r["label"] for r in rows]); idx = nn.indices.numpy()
print("1-NN accuracy over whole set (random split upper-bound hint):", (lab == lab[idx]).mean().round(3))
out = sys.argv[1]
for k, t in enumerate((0.95, 0.85, 0.75, 0.65)):
    i = int(np.argmin(np.abs(v - t))); j = idx[i]
    a, b = np.load(FRAMES / rows[i]["path"]), np.load(FRAMES / rows[j]["path"])
    cv2.imwrite(f"{out}/pair_{t}.jpg", cv2.cvtColor(np.concatenate([a[len(a)//2], b[len(b)//2]], 1), cv2.COLOR_RGB2BGR))
    print(t, rows[i]["path"], "<->", rows[j]["path"], round(float(v[i]), 3))
