"""
TASK 1 (helper) - Extract the still-unlabeled (query, creator) pairs from
task1_labeling.csv, deduplicated (the same creator often appears for the
same query across multiple configs), together with their transcription
text, for manual/assisted relevance review.

Run:
    python task1_extract_unlabeled.py
Writes unlabeled_pairs_for_review.csv (query, creator, transcription, relevant).
"""
import csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELED_CSV = os.path.join(HERE, "task1_labeling.csv")
TRANSCRIPTIONS_CSV = os.path.join(ROOT, "transcriptions", "pipeline_streaming_transcriptions.csv")
OUT_CSV = os.path.join(HERE, "unlabeled_pairs_for_review.csv")


def main():
    transcriptions = pd.read_csv(TRANSCRIPTIONS_CSV).set_index("creator")["transcription"].to_dict()

    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seen = set()
    pairs = []
    for row in rows:
        if row["relevant"].strip() != "":
            continue
        key = (row["query"], row["creator"])
        if key in seen:
            continue
        seen.add(key)
        text = str(transcriptions.get(row["creator"], "")).strip().replace("\n", " ")
        pairs.append({"query": row["query"], "creator": row["creator"], "transcription": text[:300], "relevant": ""})

    pairs.sort(key=lambda p: (p["query"], p["creator"]))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "creator", "transcription", "relevant"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Wrote {len(pairs)} unique unlabeled (query, creator) pairs to {OUT_CSV}")


if __name__ == "__main__":
    main()
