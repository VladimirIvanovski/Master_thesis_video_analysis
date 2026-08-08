"""
TASK 1 (step 2) - Best embedding combination.

Computes Precision@10 from your manually labeled task1_labeling.csv, and
builds a final comparison table that also includes your already-validated
0.85/0.15 results (approximated from the good/bad feedback already stored
in Elasticsearch's user_interactions index, collected via the Flask demo).
This lets you sanity-check that the earlier numbers are consistent with a
fresh, controlled Precision@10 measurement.

Also writes winning_config.json (best of the 5 *new* configs) for Task 2.

Run task1_generate_labeling_csv.py and label the CSV BEFORE running this.
"""
import csv
import json
import os
from collections import defaultdict

from common import CONFIGS, QUERIES, TOP_K, load_es_feedback

LABELED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_labeling.csv")
WINNER_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winning_config.json")

ES_USERNAME = "vladimir"       # username whose feedback was collected in the Flask demo
ES_EXISTING_LABEL = "85_15 (existing, from ES feedback)"


def load_labeled_precisions():
    """Returns {config_name: {query: precision@10}} from the manually labeled CSV."""
    hits = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(lambda: defaultdict(int))

    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            relevant = row["relevant"].strip()
            if relevant == "":
                raise ValueError(
                    f"Row not labeled yet (config={row['config']}, query={row['query']}, "
                    f"creator={row['creator']}). Fill in the 'relevant' column (0/1) for every row first."
                )
            config, query = row["config"], row["query"]
            counts[config][query] += 1
            hits[config][query] += int(relevant)

    precisions = defaultdict(dict)
    for config, per_query in hits.items():
        for query, n_hits in per_query.items():
            precisions[config][query] = n_hits / counts[config][query]
    return precisions


def load_existing_es_precision(username=ES_USERNAME):
    """
    Approximates Precision@10 for the currently-deployed 0.85/0.15 config from the
    good/bad creator feedback already stored in Elasticsearch (collected earlier
    via the Flask demo). precision@10 = # creators labeled "good" for that query / 10.
    """
    try:
        feedback = load_es_feedback(username)
    except Exception as e:
        print(f"Could not read Elasticsearch feedback ({e}). Skipping the existing-results row.")
        return {}

    precision = {}
    for query in QUERIES:
        n_good = len(feedback[query]["good"])
        n_labeled = n_good + len(feedback[query]["bad"])
        if n_labeled == 0:
            print(f"  no ES feedback found for '{query}', skipping it for the existing-results average")
            continue
        precision[query] = n_good / TOP_K
        print(f"  existing ES feedback for '{query}': {n_good} good / {n_labeled} labeled -> precision@10={precision[query]:.2f}")
    return precision


def print_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    for name, avg in rows:
        print(f"{name:<32} avg Precision@10 = {avg:.3f}")


def main():
    labeled_precisions = load_labeled_precisions()

    # --- Per-query Precision@10 for each of the 5 new configs ---
    print("Per-query Precision@10 (new, manually labeled):")
    print("config".ljust(12) + "".join(q.ljust(10) for q in QUERIES))
    per_config_avg = {}
    for config_name, _, _ in CONFIGS:
        per_query = labeled_precisions.get(config_name, {})
        print(config_name.ljust(12) + "".join(f"{per_query.get(q, 0):.2f}".ljust(10) for q in QUERIES))
        per_config_avg[config_name] = sum(per_query.get(q, 0) for q in QUERIES) / len(QUERIES)

    # --- Existing (already-validated) results for the current 0.85/0.15 config ---
    print("\nReading existing 0.85/0.15 validation results from Elasticsearch...")
    existing_precision = load_existing_es_precision()
    existing_avg = (sum(existing_precision.values()) / len(QUERIES)) if len(existing_precision) == len(QUERIES) else None

    # --- Final comparison table ---
    comparison_rows = [(config_name, per_config_avg[config_name]) for config_name, _, _ in CONFIGS]
    if existing_avg is not None:
        comparison_rows.append((ES_EXISTING_LABEL, existing_avg))
    else:
        print("(Not enough ES feedback for all 5 queries - existing-results row omitted from the table.)")
    print_table("FINAL COMPARISON: config vs. average Precision@10", comparison_rows)

    # --- Pick the winner among the 5 *new* configs and hand it off to Task 2 ---
    winner_name, winner_avg = max(((c, per_config_avg[c]) for c, _, _ in CONFIGS), key=lambda x: x[1])
    winner_img_w, winner_txt_w = next((iw, tw) for c, iw, tw in CONFIGS if c == winner_name)
    with open(WINNER_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": winner_name,
                "image_weight": winner_img_w,
                "text_weight": winner_txt_w,
                "avg_precision_at_10": winner_avg,
            },
            f,
            indent=2,
        )

    print(f"\nWinning config: {winner_name} (image={winner_img_w}, text={winner_txt_w}, avg P@10={winner_avg:.3f})")
    print(f"Saved to {WINNER_JSON} for Task 2.")


if __name__ == "__main__":
    main()
