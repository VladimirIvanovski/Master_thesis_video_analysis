"""
TASK 1 (optional helper) - Auto-fill task1_labeling.csv from existing feedback.

Uses the good/bad creator labels you already saved in Elasticsearch's
user_interactions index (via the Flask demo apps) to pre-fill the 'relevant'
column in task1_labeling.csv:
  - creator marked "good" for that query      -> relevant = 1
  - creator marked "bad"  for that query       -> relevant = 0
  - creator marked BOTH good and bad (conflict) -> left blank, reported
  - creator never labeled for that query        -> left blank, needs manual review

Run this AFTER task1_generate_labeling_csv.py and BEFORE task1_compute_precision.py:
    python task1_autolabel_from_es.py
Then manually fill in whatever rows are still blank.
"""
import csv
import os

from common import QUERIES, load_es_feedback

LABELED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_labeling.csv")
ES_USERNAME = "vladimir"


def main():
    feedback = load_es_feedback(ES_USERNAME)
    for query in QUERIES:
        print(f"  {query:<8} -> {len(feedback[query]['good'])} good / {len(feedback[query]['bad'])} bad labeled in ES")

    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = f.fieldnames if hasattr(f, "fieldnames") else list(rows[0].keys())

    n_filled_good, n_filled_bad, n_conflict, n_unlabeled = 0, 0, 0, 0
    for row in rows:
        if row["relevant"].strip() != "":
            continue  # don't overwrite anything already labeled by hand

        creator = row["creator"]
        good = feedback[row["query"]]["good"]
        bad = feedback[row["query"]]["bad"]
        in_good, in_bad = creator in good, creator in bad

        if in_good and in_bad:
            n_conflict += 1
            print(f"  CONFLICT: '{creator}' is labeled both good and bad for '{row['query']}' - left blank")
        elif in_good:
            row["relevant"] = "1"
            n_filled_good += 1
        elif in_bad:
            row["relevant"] = "0"
            n_filled_bad += 1
        else:
            n_unlabeled += 1

    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAuto-filled {n_filled_good} rows as relevant=1, {n_filled_bad} rows as relevant=0.")
    print(f"{n_conflict} conflicting rows and {n_unlabeled} never-labeled rows still need manual review.")
    print(f"Open {LABELED_CSV} and fill in the remaining blank 'relevant' cells, then run task1_compute_precision.py")


if __name__ == "__main__":
    main()
