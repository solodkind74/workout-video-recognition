"""Webcam: live exercise recognition, or record/score your own clips (real-world test set, never trained on).

uv run python cam.py                         # live: top-3 overlay, q to quit
uv run python cam.py --test squat -n 5       # record 5 clips of 4 s -> data/mine/squat/*.npy, then score
uv run python cam.py --score                 # re-score everything in data/mine/
Stand back so your whole body is in frame.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from common import CLASSES, FPS, ROOT, square
from features_video import load as load_vmae, run as run_vmae, windows
from train_vmae import make_head

MINE = ROOT / "data/mine"


def load_head(run):
    ck = torch.load(f"{run}/model.pt")
    h = make_head(768, len(CLASSES), ck["head"])
    h.load_state_dict(ck["state"])
    return h.eval()


@torch.no_grad()
def predict(vmae, head, frames, dev):
    f, _ = run_vmae(vmae, np.stack([frames[w] for w in windows(len(frames))]), dev)
    return head(f).softmax(-1).mean(0)


def frames_from_cam(cap, seconds, show=None):
    out, t0, nxt = [], time.time(), 0.0
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if not ok:
            continue
        if time.time() - t0 >= nxt:
            out.append(square(f)); nxt += 1 / FPS
        if show:
            cv2.putText(f, show(time.time() - t0), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("cam", f); cv2.waitKey(1)
    return np.stack(out)


def open_cam(index=None):
    """First camera that shows a picture. Linked iPhones (Continuity Camera) often take index 0 and send black frames."""
    for i in [index] if index is not None else range(4):
        cap = cv2.VideoCapture(i)
        deadline = time.time() + 3  # macOS cameras return empty or black frames for a moment after opening
        while cap.isOpened() and time.time() < deadline:
            ok, f = cap.read()
            if ok and f.mean() > 5:
                print(f"using camera {i}")
                return cap
        cap.release()
    return None


def score_dir(vmae, head, dev):
    hits = total = 0
    MINE.mkdir(parents=True, exist_ok=True)
    for d in sorted(MINE.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npy")):
            p = predict(vmae, head, np.load(f), dev)
            top = p.topk(3)
            total += 1; hits += CLASSES[top.indices[0]] == d.name
            print(f"{d.name:22s} {f.stem:>3s} -> " + ", ".join(f"{CLASSES[i]} {v:.2f}" for v, i in zip(top.values, top.indices)))
    print(f"top-1: {hits}/{total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/vmae_linear_group")
    ap.add_argument("--test", choices=CLASSES)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=4)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cam", type=int, help="camera index (default: first one that shows a picture)")
    a = ap.parse_args()
    dev = "mps"
    vmae, head = load_vmae(dev), load_head(a.run)
    if a.score:
        score_dir(vmae, head, dev)
        raise SystemExit
    cap = open_cam(a.cam)
    if cap is None:
        raise SystemExit("No camera with a picture. Allow your terminal app in System Settings > Privacy & Security > "
                         "Camera, then quit the terminal app completely (Cmd+Q), reopen it and try again.")
    if a.test:
        d = MINE / a.test
        d.mkdir(parents=True, exist_ok=True)
        start = len(list(d.glob("*.npy")))
        for k in range(a.n):
            frames_from_cam(cap, 3, lambda t: f"get ready: {a.test} {k + 1}/{a.n} in {3 - t:.0f}")
            clip = frames_from_cam(cap, a.seconds, lambda t: f"REC {t:.1f}s")
            np.save(d / f"{start + k}.npy", clip)
            p = predict(vmae, head, clip, dev)
            print(f"clip {start + k}: {CLASSES[p.argmax()]} ({p.max():.2f})")
        raise SystemExit
    buf, last, last_pred, probs = deque(maxlen=16), 0.0, 0.0, deque(maxlen=3)
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        now = time.time()
        if now - last >= 1 / FPS:
            buf.append(square(f)); last = now
        if len(buf) == 16 and now - last_pred >= 0.5:  # ~2 predictions/s, smoothed over the last 3
            last_pred = now
            probs.append(predict(vmae, head, np.stack(buf), dev))
        if probs:
            p = torch.stack(list(probs)).mean(0)
            for r, (v, i) in enumerate(zip(*p.topk(3))):
                cv2.putText(f, f"{CLASSES[i]} {v:.2f}", (30, 60 + 50 * r), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 255, 0) if r == 0 else (200, 200, 200), 3)
        cv2.imshow("cam", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
