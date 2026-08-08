"""
TASK 1 (step 1) - Best embedding combination.

Builds a FAISS index for each of the 5 visual/text weight configs, runs the
5 test queries against each (25 searches total), and exports the top-10
results to a CSV for manual relevance labeling.

Run this first:
    python task1_generate_labeling_csv.py

Then open task1_labeling.csv and fill the "relevant" column with 1
(relevant) or 0 (not relevant) for every row, and finally run:
    python task1_compute_precision.py
"""
import csv
import os

import numpy as np

from common import (
    CONFIGS,
    QUERIES,
    TOP_K,
    build_combined_embeddings,
    build_faiss_index,
    embed_query,
    load_creators_and_embeddings,
)

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_labeling.csv")


def main():
    creators, img_embs, txt_embs = load_creators_and_embeddings()
    print(f"Loaded {len(creators)} creators.")

    rows = []
    for config_name, img_w, txt_w in CONFIGS:
        combined = build_combined_embeddings(img_embs, txt_embs, img_w, txt_w)
        index = build_faiss_index(combined)

        for query in QUERIES:
            query_vec = embed_query(query)
            scores, idxs = index.search(np.expand_dims(query_vec, axis=0), TOP_K)

            for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
                rows.append({
                    "config": config_name,
                    "image_weight": img_w,
                    "text_weight": txt_w,
                    "query": query,
                    "rank": rank,
                    "creator": creators[idx],
                    "score": round(float(score), 4),
                    "relevant": "",  # fill in manually: 1 = relevant, 0 = not relevant
                })
            print(f"  {config_name:>11} | {query:<8} -> top {TOP_K} retrieved")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config", "image_weight", "text_weight", "query", "rank", "creator", "score", "relevant"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows ({len(CONFIGS)} configs x {len(QUERIES)} queries x {TOP_K} results) to {OUT_CSV}")
    print("Next: label the 'relevant' column (0/1) for every row, then run task1_compute_precision.py")


if __name__ == "__main__":
    main()
