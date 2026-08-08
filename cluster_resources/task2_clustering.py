"""
TASK 2 - Clustering quality.

Uses the winning embedding config selected by task1_compute_precision.py
(winning_config.json):
  1. Agglomerative hierarchical clustering (cosine distance) on all creator
     embeddings, testing k=3..15 clusters, scored with silhouette score.
  2. Reports which creators fall into which cluster at the best k.
  3. For each of the 5 test queries, searches with the winning config and
     reports what % of the top-10 results fall into the single most common
     ("same") cluster.

Run this AFTER task1_generate_labeling_csv.py + task1_compute_precision.py:
    python task2_clustering.py
"""
import csv
import json
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from common import QUERIES, TOP_K, build_combined_embeddings, build_faiss_index, embed_query, load_creators_and_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
WINNER_JSON = os.path.join(HERE, "winning_config.json")
CLUSTERS_CSV = os.path.join(HERE, "task2_creator_clusters.csv")

K_RANGE = range(3, 16)  # 3..15 inclusive

# Agglomerative clustering + silhouette score tends to reward splitting off a
# single far-away outlier into its own "cluster", since an isolated point
# trivially gets a near-perfect silhouette. That's a statistical artifact,
# not a meaningful grouping, so when picking the *best* k we require every
# cluster to have at least MIN_CLUSTER_SIZE creators. The full silhouette
# table below still reports every k (including degenerate ones) for transparency.
MIN_CLUSTER_SIZE = 5

# Average-linkage cosine clustering chains outliers off one at a time (every
# k from 3-15 has a singleton cluster here) instead of finding real groups.
# Ward linkage minimizes within-cluster variance and avoids that chaining,
# but scikit-learn only supports Ward with Euclidean distance. Since our
# embeddings are unit-normalized, Euclidean and cosine distance are monotonic
# in one another (||u-v||^2 = 2*(1 - cos_sim(u,v))), so clustering with Ward +
# Euclidean on unit vectors is a standard, equivalent way to do cosine-based
# hierarchical clustering while still scoring/reporting silhouette with the
# requested cosine metric.
CLUSTER_METRIC = "euclidean"
CLUSTER_LINKAGE = "ward"


def load_winning_weights():
    with open(WINNER_JSON, encoding="utf-8") as f:
        winner = json.load(f)
    print(
        f"Using winning config from Task 1: {winner['config']} "
        f"(image={winner['image_weight']}, text={winner['text_weight']}, "
        f"avg P@10={winner['avg_precision_at_10']:.3f})"
    )
    return winner["image_weight"], winner["text_weight"]


def cluster(embeddings, k):
    return AgglomerativeClustering(n_clusters=k, metric=CLUSTER_METRIC, linkage=CLUSTER_LINKAGE).fit_predict(embeddings)


def find_best_k(embeddings):
    """Tries k=3..15 agglomerative clusters, returns the full
    (k, silhouette, min_cluster_size) table plus the best *non-degenerate* k
    (i.e. every cluster has >= MIN_CLUSTER_SIZE creators). Silhouette is
    always reported using cosine distance, as requested."""
    scores = []
    for k in K_RANGE:
        labels = cluster(embeddings, k)
        score = silhouette_score(embeddings, labels, metric="cosine")
        min_size = np.bincount(labels).min()
        scores.append((k, score, min_size))
        flag = "" if min_size >= MIN_CLUSTER_SIZE else "  (degenerate: smallest cluster has %d creator%s)" % (
            min_size, "" if min_size == 1 else "s",
        )
        print(f"  k={k:>2}  silhouette={score:.4f}  smallest_cluster={min_size:>3}{flag}")

    valid = [(k, s, m) for k, s, m in scores if m >= MIN_CLUSTER_SIZE]
    if not valid:
        print(f"\n  No k in range had every cluster >= {MIN_CLUSTER_SIZE} creators; falling back to raw best silhouette.")
        valid = scores
    best_k, best_score, _ = max(valid, key=lambda x: x[1])
    return scores, best_k, best_score


def main():
    img_w, txt_w = load_winning_weights()
    creators, img_embs, txt_embs = load_creators_and_embeddings()
    embeddings = build_combined_embeddings(img_embs, txt_embs, img_w, txt_w)

    print("\nSilhouette scores per cluster count (k=3..15):")
    scores, best_k, best_score = find_best_k(embeddings)
    print(f"\nBest k = {best_k} (silhouette = {best_score:.4f})")

    # --- Final clustering at best_k, creator -> cluster assignment ---
    labels = cluster(embeddings, best_k)

    with open(CLUSTERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["creator", "cluster_id"])
        for creator, cluster_id in zip(creators, labels):
            writer.writerow([creator, int(cluster_id)])
    print(f"Wrote creator -> cluster assignments to {CLUSTERS_CSV}")

    # --- % of top-10 search results landing in the same (dominant) cluster ---
    index = build_faiss_index(embeddings)
    print("\n% of top-10 results in the same cluster, per query:")
    for query in QUERIES:
        query_vec = embed_query(query)
        _, idxs = index.search(np.expand_dims(query_vec, axis=0), TOP_K)
        result_clusters = labels[idxs[0]]
        dominant_cluster = np.bincount(result_clusters).argmax()
        pct_same = 100.0 * np.sum(result_clusters == dominant_cluster) / TOP_K
        print(f"  {query:<8} -> dominant cluster={dominant_cluster}, {pct_same:.0f}% of top-10 share it")


if __name__ == "__main__":
    main()
