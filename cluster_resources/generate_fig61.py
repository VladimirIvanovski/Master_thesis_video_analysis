"""
Generates the previously-missing Fig. 6.1 (Speedup vs. number of CPU workers),
from Table 6.2's data (20 creators / 50 videos scalability benchmark).
"""
import os

import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_figures")

WORKERS = [2, 4, 8, 12]
SPEEDUP = [1.00, 1.21, 1.30, 1.30]
IDEAL = [w / WORKERS[0] for w in WORKERS]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(WORKERS, SPEEDUP, marker="o", color="#3B6FA0", linewidth=2, label="Measured speedup")
ax.plot(WORKERS, IDEAL, linestyle="--", color="#888888", linewidth=1.5, label="Ideal linear scaling")
ax.set_xlabel("Number of CPU workers")
ax.set_ylabel("Speedup (relative to 2 workers)")
ax.set_title("Speedup vs. number of CPU workers")
ax.set_xticks(WORKERS)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_speedup.png"), dpi=150)
print("Saved fig_speedup.png")
