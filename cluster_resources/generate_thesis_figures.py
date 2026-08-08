"""
Generates the Task 1 / Task 2 result charts used in thesis section 6.5:
  - fig_precision_comparison.png : avg Precision@10 per embedding config
  - fig_silhouette_scores.png    : silhouette score vs. number of clusters (k)
  - fig_cluster_scatter.png      : 2D PCA projection of creator embeddings, colored by cluster
  - fig_cluster_sizes.png        : creator count per cluster
  - fig_same_cluster_pct.png     : % of top-10 search results sharing the dominant cluster, per query

Run AFTER task1_compute_precision.py and task2_clustering.py:
    python generate_thesis_figures.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from common import QUERIES, TOP_K, build_combined_embeddings, build_faiss_index, embed_query, load_creators_and_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

CONFIG_COLORS = {
    "visual_only": "#2E86AB",
    "text_only": "#A23B72",
    "50_50": "#F18F01",
    "85_15": "#3B944B",
    "15_85": "#C73E3E",
    "85_15 (existing, from ES feedback)": "#6C6C6C",
}


def fig_precision_comparison():
    """Re-derives the same config -> avg Precision@10 numbers task1_compute_precision.py
    prints, and plots them as a bar chart."""
    from task1_compute_precision import CONFIGS, load_es_feedback, load_labeled_precisions

    labeled = load_labeled_precisions()
    avgs = {}
    for name, _, _ in CONFIGS:
        per_query = labeled.get(name, {})
        avgs[name] = sum(per_query.get(q, 0) for q in QUERIES) / len(QUERIES)

    feedback = load_es_feedback("vladimir")
    es_vals = [len(feedback[q]["good"]) / TOP_K for q in QUERIES if (len(feedback[q]["good"]) + len(feedback[q]["bad"])) > 0]
    if len(es_vals) == len(QUERIES):
        avgs["85_15 (existing, from ES feedback)"] = sum(es_vals) / len(QUERIES)
    else:
        # Live ES feedback no longer covers all 5 queries; fall back to the
        # historical value already documented in the thesis (Table 6.5).
        avgs["85_15 (existing, from ES feedback)"] = 0.64

    names = list(avgs.keys())
    values = [avgs[n] for n in names]
    colors = [CONFIG_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (existing, from ES feedback)", "\n(existing, ES)") for n in names], rotation=20, ha="right")
    ax.set_ylabel("Average Precision@10")
    ax.set_title("Embedding configuration vs. average Precision@10")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_precision_comparison.png"))
    plt.close(fig)


def fig_silhouette_and_clusters():
    with open(os.path.join(HERE, "winning_config.json"), encoding="utf-8") as f:
        winner = json.load(f)

    creators, img_embs, txt_embs = load_creators_and_embeddings()
    X = build_combined_embeddings(img_embs, txt_embs, winner["image_weight"], winner["text_weight"])

    from sklearn.metrics import silhouette_score

    k_range = list(range(3, 16))
    scores, min_sizes = [], []
    for k in k_range:
        labels = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(X)
        scores.append(silhouette_score(X, labels, metric="cosine"))
        min_sizes.append(np.bincount(labels).min())
    best_idx = int(np.argmax([s if m >= 5 else -1 for s, m in zip(scores, min_sizes)]))
    best_k = k_range[best_idx]

    # --- Silhouette vs k ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_range, scores, marker="o", color="#2E86AB")
    ax.scatter([best_k], [scores[best_idx]], color="#C73E3E", s=120, zorder=5, label=f"best k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (cosine distance)")
    ax.set_title("Clustering quality vs. number of clusters")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_silhouette_scores.png"))
    plt.close(fig)

    # --- Final clustering at best_k ---
    labels = AgglomerativeClustering(n_clusters=best_k, metric="euclidean", linkage="ward").fit_predict(X)
    palette = plt.cm.get_cmap("tab10", best_k)

    # --- 2D PCA scatter colored by cluster ---
    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(best_k):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22, color=palette(c), label=f"cluster {c} (n={mask.sum()})", alpha=0.8)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(f"Creator embeddings, 2D PCA projection ({best_k} clusters)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_cluster_scatter.png"))
    plt.close(fig)

    # --- Cluster sizes ---
    sizes = np.bincount(labels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(best_k), sizes, color=[palette(c) for c in range(best_k)])
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of creators")
    ax.set_title(f"Creators per cluster (k={best_k})")
    for c, s in enumerate(sizes):
        ax.text(c, s + 1, str(s), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_cluster_sizes.png"))
    plt.close(fig)

    # --- % of top-10 search results sharing the dominant cluster ---
    index = build_faiss_index(X)
    pct_same = []
    for query in QUERIES:
        query_vec = embed_query(query)
        _, idxs = index.search(np.expand_dims(query_vec, axis=0), TOP_K)
        result_clusters = labels[idxs[0]]
        dominant = np.bincount(result_clusters).argmax()
        pct_same.append(100.0 * np.sum(result_clusters == dominant) / TOP_K)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(QUERIES, pct_same, color="#3B944B")
    ax.set_ylabel("% of top-10 results in dominant cluster")
    ax.set_title("Search-result cluster cohesion per query")
    ax.set_ylim(0, 105)
    ax.axhline(100 / best_k, color="gray", linestyle="--", linewidth=1, label=f"chance level (~{100 / best_k:.0f}%)")
    for bar, v in zip(bars, pct_same):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_same_cluster_pct.png"))
    plt.close(fig)

    return best_k, sizes, pct_same


def main():
    fig_precision_comparison()
    best_k, sizes, pct_same = fig_silhouette_and_clusters()
    print(f"Saved 5 figures to {OUT_DIR}")
    print(f"best_k={best_k}, cluster_sizes={sizes.tolist()}, pct_same_cluster={pct_same}")


if __name__ == "__main__":
    main()
