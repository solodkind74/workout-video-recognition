import cv2, sys, glob, os, collections
root = sys.argv[1]
stats = []
for f in sorted(glob.glob(os.path.join(root, "*", "*"))):
    c = cv2.VideoCapture(f)
    n, fps = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS)
    w, h = c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT)
    stats.append((os.path.basename(os.path.dirname(f)), n / max(fps, 1), fps, int(w), int(h)))
import statistics as s
d = [x[1] for x in stats]
print("videos", len(stats), "dur median %.1fs min %.1f max %.1f total %.0f min" % (s.median(d), min(d), max(d), sum(d) / 60))
print("fps", collections.Counter(round(x[2]) for x in stats).most_common(5))
print("res", collections.Counter((x[3], x[4]) for x in stats).most_common(8))
print("portrait", sum(x[4] > x[3] for x in stats))
