"""
Generates Figure 4.1: high-level system architecture diagram (colorful,
boxes + arrows), showing the full pipeline from TikTok download through
the Flask demo application. Fills the previously-empty Figure 4.1 slot in
the thesis.

Run:
    python generate_architecture_diagram.py
Writes thesis_figures/fig_architecture.png
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "thesis_figures", "fig_architecture.png")


def box(ax, xy, w, h, title, subtitle, color):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 linewidth=1.4, edgecolor="#333333", facecolor=color))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8.3, color="white")


def arrow(ax, start, end, color="#333333"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                  linewidth=1.6, color=color, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    w, h, y = 2.0, 1.1, 4.6

    box(ax, (0.3, y), w, h, "Stage 0", "TikTok download\n(yt-dlp)", "#5B7DB1")
    box(ax, (2.7, y), w, h, "Stage 1", "Media extraction\n(FFmpeg: PCM audio + JPEG)", "#E07B39")
    box(ax, (5.1, y), w, h, "Stage 2", "Transcription\n(faster-whisper, GPU)", "#3B944B")
    box(ax, (7.5, y), w, h, "Stage 3", "Embedding\n(OpenCLIP ViT-B/32, GPU)", "#8E5DB0")

    arrow(ax, (2.3, y + h / 2), (2.7, y + h / 2))
    arrow(ax, (4.7, y + h / 2), (5.1, y + h / 2))
    arrow(ax, (7.1, y + h / 2), (7.5, y + h / 2))

    # Ray orchestration box around Stage 1-3
    ax.add_patch(Rectangle((2.55, y - 0.35), 5.15, h + 0.7, linewidth=1.6,
                            edgecolor="#B0301E", facecolor="none", linestyle="--"))
    ax.text(5.1, y + h + 0.5, "Ray: streaming producer-consumer pipeline (bounded memory, zero disk I/O)",
            ha="center", fontsize=9, color="#B0301E", fontweight="bold")

    y2 = 2.4
    box(ax, (7.0, y2), w, h, "Vector Index", "FAISS IndexFlatIP\n(combined embeddings)", "#C73E3E")
    box(ax, (9.4, y2), w, h, "Metadata Store", "Elasticsearch\n(transcripts, frames, vectors)", "#2E86AB")

    arrow(ax, (8.3, y), (8.0, y2 + h))
    arrow(ax, (9.0, y), (9.9, y2 + h))

    y3 = 0.3
    box(ax, (5.6, y3), 2.8, h, "Flask Demo App", "Semantic + personalized search\n(niche queries, like/dislike feedback)", "#4A4A4A")

    arrow(ax, (7.5, y2), (7.3, y3 + h))
    arrow(ax, (10.0, y2), (7.7, y3 + h))

    ax.set_title("High-level system architecture", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
