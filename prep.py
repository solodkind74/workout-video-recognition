"""Decode every video once -> data/frames/<class>/<name>.npy + data/index.csv"""
import csv
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from common import CLASSES, FRAMES, RAW, read_video


def job(args):
    cls, f = args
    out = FRAMES / cls / (f.stem + ".npy")
    if not out.exists():
        v = read_video(f)
        if v is None:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, v)
    return cls, str(out.relative_to(FRAMES)), len(np.load(out, mmap_mode="r"))


if __name__ == "__main__":
    jobs = [(c, f) for c in CLASSES for f in sorted((RAW / c).iterdir()) if f.suffix.lower() in (".mp4", ".mov")]
    with Pool(8) as p:
        rows = [r for r in tqdm(p.imap_unordered(job, jobs), total=len(jobs)) if r]
    rows.sort()
    with open(FRAMES.parent / "index.csv", "w") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "path", "frames"])
        w.writerows(rows)
    print(len(rows), "videos,", sum(r[2] for r in rows), "frames;", len(jobs) - len(rows), "failed")
