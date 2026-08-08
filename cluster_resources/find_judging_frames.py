import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results_4")

creators = sorted({
    r["creator"]
    for r in csv.DictReader(open(os.path.join(ROOT, "cluster_resources", "unlabeled_pairs_for_review.csv"), encoding="utf-8"))
})

out = []
for c in creators:
    cdir = os.path.join(RESULTS_DIR, c)
    frames = sorted(glob.glob(os.path.join(cdir, "*", "frames", "*_05.jpg")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(cdir, "*", "frames", "*.jpg")))
    out.append((c, frames[0] if frames else ""))

with open(os.path.join(ROOT, "cluster_resources", "judging_frame_paths.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["creator", "frame_path"])
    w.writerows(out)

missing = [c for c, p in out if not p]
print(f"{len(out)} creators, {len(missing)} missing frames")
print("missing:", missing)
