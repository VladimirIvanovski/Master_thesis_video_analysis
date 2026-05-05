"""
plot_results.py — Generate presentation chart from scalability_results.csv
Run: python plot_results.py  →  saves scalability_chart.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("scalability_results.csv")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "text.color":        "#e6edf3",
    "font.family":       "monospace",
})

COLORS   = {"10": "#58a6ff", "20": "#3fb950", "30": "#f78166"}
WORKERS  = [1, 2, 4, 8]
VIDEO_NS = [10, 20, 30]

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(
    "TikTok Video Pipeline at Scale — Ray Distributed Benchmark",
    fontsize=14, fontweight="bold", color="#e6edf3", y=1.01
)

# ── Panel 1: Speedup curve ─────────────────────────────────────────────────────
ax = axes[0]
ax.set_title("Speedup vs CPU Workers", fontsize=11, pad=10)

# Ideal linear speedup reference
ax.plot(WORKERS, WORKERS, "--", color="#30363d", linewidth=1.2, label="Ideal (linear)")

for n in VIDEO_NS:
    sub = df[df["videos"] == n].sort_values("cpu_workers")
    ax.plot(sub["cpu_workers"], sub["speedup"],
            marker="o", linewidth=2, markersize=7,
            color=COLORS[str(n)], label=f"{n} videos")
    # Annotate last point
    last = sub.iloc[-1]
    ax.annotate(f"×{last['speedup']}",
                xy=(last["cpu_workers"], last["speedup"]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=8, color=COLORS[str(n)])

ax.set_xlabel("CPU Workers")
ax.set_ylabel("Speedup (vs 1 worker)")
ax.set_xticks(WORKERS)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_ylim(0.8, max(WORKERS) + 0.5)


# ── Panel 2: Total wall time bars ─────────────────────────────────────────────
ax = axes[1]
ax.set_title("Total Wall Time per Run", fontsize=11, pad=10)

x      = np.arange(len(WORKERS))
width  = 0.25
offsets = [-width, 0, width]

for i, n in enumerate(VIDEO_NS):
    sub = df[df["videos"] == n].sort_values("cpu_workers")
    bars = ax.bar(x + offsets[i], sub["total_wall_s"].values,
                  width=width - 0.02, color=COLORS[str(n)],
                  label=f"{n} videos", alpha=0.85)
    # Label bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f"{h:.0f}s", ha="center", va="bottom", fontsize=7, color="#8b949e")

ax.set_xlabel("CPU Workers")
ax.set_ylabel("Total Time (seconds)")
ax.set_xticks(x)
ax.set_xticklabels(WORKERS)
ax.legend(fontsize=8)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# ── Panel 3: Stage 2 vs Stage 3 breakdown (stacked, 30 videos) ───────────────
ax = axes[2]
ax.set_title("Stage Breakdown — 30 Videos\n(Stage2=Whisper/CPU  Stage3=CLIP/GPU)", fontsize=10, pad=10)

sub30 = df[df["videos"] == 30].sort_values("cpu_workers")

# Stage 2 total ≈ total_wall - stage3 sequential time
# Show avg per-video times
s2 = sub30["stage2_avg_s"].values
s3 = sub30["stage3_avg_s"].values
x3 = np.arange(len(WORKERS))

b1 = ax.bar(x3, s2, color="#58a6ff", label="Stage 2 avg/video (Whisper CPU)", alpha=0.85)
b2 = ax.bar(x3, s3, bottom=s2, color="#bc8cff", label="Stage 3 avg/creator (CLIP GPU)", alpha=0.85)

for bar, v in zip(b1, s2):
    ax.text(bar.get_x() + bar.get_width()/2, v/2,
            f"{v:.1f}s", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
for bar, v, base in zip(b2, s3, s2):
    ax.text(bar.get_x() + bar.get_width()/2, base + v/2,
            f"{v:.2f}s", ha="center", va="center", fontsize=8, color="white")

ax.set_xlabel("CPU Workers")
ax.set_ylabel("Time (seconds)")
ax.set_xticks(x3)
ax.set_xticklabels(WORKERS)
ax.legend(fontsize=8)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)

# ── Footer note ───────────────────────────────────────────────────────────────
fig.text(0.5, -0.03,
    "Hardware: 1× laptop GPU (CUDA)  |  CPU workers simulate distributed Ray nodes  |  "
    "Stage 2 & 3 run concurrently via ray.wait()",
    ha="center", fontsize=8, color="#8b949e")

plt.tight_layout()
plt.savefig("scalability_chart.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print("Saved: scalability_chart.png")
