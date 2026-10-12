"""Shared paths, class list and frame helpers."""
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
RAW = ROOT / "data/kaggle/datasets/hasyimabdillah/workoutfitness-video/versions/5"
FRAMES = ROOT / "data/frames"  # one uint8 .npy (T, 224, 224, 3) per video
CLASSES = sorted(os.listdir(RAW)) if RAW.exists() else []
SIZE = 224
FPS = 8  # store at ~8 fps (VideoMAE-Kinetics sees 16 frames x stride 4 @ 30fps ~ 7.5 fps)
MAX_T = 80  # 10 s


def square(frame_bgr, size=SIZE):
    """Short side -> size, center crop, BGR -> RGB."""
    h, w = frame_bgr.shape[:2]
    s = size / min(h, w)
    f = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = f.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return cv2.cvtColor(f[y:y + size, x:x + size], cv2.COLOR_BGR2RGB)


def read_video(path, fps=FPS, max_t=MAX_T):
    """Decode at ~fps, keep the middle max_t frames."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(src / fps, 1)
    frames, i, nxt = [], 0, 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i >= nxt:
            ok, f = cap.retrieve()
            if ok:
                frames.append(square(f))
            nxt += step
        i += 1
    if len(frames) > max_t:
        s = (len(frames) - max_t) // 2
        frames = frames[s:s + max_t]
    return np.stack(frames) if frames else None


def sample_clip(n, t=16, rng=None):
    """Indices of t frames from a video of n frames: random window if rng, else centered. Loops short videos."""
    if n >= t:
        s = rng.integers(0, n - t + 1) if rng is not None else (n - t) // 2
        return np.arange(s, s + t)
    return np.arange(t) % n
