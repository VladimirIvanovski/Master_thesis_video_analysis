"""
TASK 1 (helper) - Default any still-blank 'relevant' cells in task1_labeling.csv
to 0 (not relevant), so precision can be computed without full manual labeling.

NOTE: this is an approximation. Rows defaulted to 0 were never actually
judged (no ES feedback existed and their transcriptions were uninformative
Whisper hallucinations) - it likely undercounts precision for whichever
configs the current 0.85/0.15 feedback never covered.

Run:
    python task1_default_unlabeled.py
"""
import csv
import os

LABELED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_labeling.csv")


def main():
    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    n_defaulted = 0
    for row in rows:
        if row["relevant"].strip() == "":
            row["relevant"] = "0"
            n_defaulted += 1

    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Defaulted {n_defaulted} unlabeled rows to relevant=0 in {LABELED_CSV}")


if __name__ == "__main__":
    main()
