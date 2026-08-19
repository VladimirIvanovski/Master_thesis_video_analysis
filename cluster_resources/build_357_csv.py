"""Build the 357-creator supplementary CSV from tiktok_profile_5k.csv."""
import csv
from pathlib import Path

ROOT = Path(r"c:\Users\vladimir\PyCharmMiscProject")
SRC = Path(r"C:\Users\vladimir\Downloads\tiktok_profile_5k.csv")
TRANS = ROOT / "transcriptions" / "pipeline_streaming_transcriptions.csv"
OUT = ROOT / "cluster_resources" / "thesis_357_creators.csv"

creators = []
with TRANS.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        name = (row.get("creator") or "").strip()
        if name:
            creators.append(name)

by_lower = {}
with SRC.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        u = (row.get("username") or "").strip()
        if u:
            by_lower[u.lower()] = row

matched = []
missing = []
for name in creators:
    row = by_lower.get(name.lower())
    if row is None:
        missing.append(name)
    else:
        matched.append(row)

out_fields = list(fields)
if "tiktok_url" not in out_fields:
    out_fields.append("tiktok_url")

with OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
    w.writeheader()
    for row in matched:
        out = dict(row)
        user = (row.get("username") or "").strip()
        out["tiktok_url"] = f"https://www.tiktok.com/@{user}"
        w.writerow(out)

print("source_creators", len(creators))
print("matched", len(matched))
print("missing", len(missing))
if missing:
    print("missing names:", missing[:20])
print("wrote", OUT, "bytes", OUT.stat().st_size)
print("columns", out_fields)
