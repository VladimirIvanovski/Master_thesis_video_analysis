"""
large_scale_benchmark.py  —  Large-Scale Single Run
====================================================
Step 0: If total videos < 1000, downloads new creators from CSV
        (skips already-downloaded usernames, relaxed filter).
Step 1: Processes all available videos with 8 CPU workers + 1 GPU.
Stage1=FFmpeg/CPU  Stage2=Whisper/GPU  Stage3=CLIP/GPU  (streaming)

Run:    python large_scale_benchmark.py
Output: large_scale_results.csv + large_scale_chart.png
"""

import os, glob, time, shutil, threading, collections
import numpy as np, pandas as pd, ray, psutil, torch
from tqdm import tqdm
from config import CSV_PATH, MAX_VIDEOS_PER_CREATOR
from stage1_download_extract import process_creator

# Reuse actors from full_pipeline_benchmark
from full_pipeline_benchmark import (
    stage1_extract, GPUWhisperActor, GPUCLIPActor,
    start_web, upd, gst,
)

NUM_WORKERS = 8
RESULTS_DIR = "results_4"
OUT_CSV     = "large_scale_results.csv"
OUT_CHART   = "large_scale_chart.png"


def run_large(mp4_list: list, whisper, clip) -> dict:
    n_videos   = len(mp4_list)
    creator_of = {p: os.path.basename(os.path.dirname(p)) for p in mp4_list}
    unique_creators = list(dict.fromkeys(creator_of.values()))   # preserve order
    expected = collections.Counter(creator_of.values())

    upd(run_label=f"{n_videos} videos  x  {NUM_WORKERS} CPU workers  +  1 GPU",
        run_i=1, total_runs=1,
        s1_done=0, s1_total=n_videos,
        s2_done=0, s2_total=n_videos,
        s3_done=0, s3_total=len(unique_creators))

    bench_dir = os.path.join(RESULTS_DIR, "_bench_tmp")
    os.makedirs(bench_dir, exist_ok=True)

    # Submit all Stage-1 tasks
    s1_futures = {}
    for mp4 in mp4_list:
        creator = creator_of[mp4]
        out_dir = os.path.join(bench_dir, creator,
                               os.path.splitext(os.path.basename(mp4))[0])
        fut = stage1_extract.remote(mp4, out_dir)
        s1_futures[fut] = mp4

    creator_frames: dict[str, list] = {c: [] for c in unique_creators}
    creator_texts:  dict[str, list] = {c: [] for c in unique_creators}
    s1_times, s2_times, s3_times = [], [], []
    throughput_log = []          # (wall_time, videos_done)
    s2_futures, s3_futures = {}, {}

    pbar1 = tqdm(total=n_videos,  desc="Stage1 FFmpeg/CPU", unit="vid",
                 ncols=70, colour="blue")
    pbar2 = tqdm(total=n_videos,  desc="Stage2 Whisper/GPU", unit="vid",
                 ncols=70, colour="magenta")
    pbar3 = tqdm(total=len(unique_creators), desc="Stage3 CLIP/GPU",
                 unit="creator", ncols=70, colour="green")

    wall_start    = time.time()
    remaining_s1  = list(s1_futures.keys())
    remaining_s2  = []

    while remaining_s1 or remaining_s2 or s3_futures:
        if remaining_s1:
            done1, remaining_s1 = ray.wait(remaining_s1, num_returns=1, timeout=0.1)
            for f in done1:
                r1 = ray.get(f)
                s1_times.append(r1["s1_time"])
                creator = r1["creator"]
                creator_frames[creator].extend(r1["frames"])
                s2f = whisper.transcribe.remote(r1["wav"])
                s2_futures[s2f] = creator
                remaining_s2.append(s2f)
                pbar1.update(1)
                upd(s1_done=pbar1.n)
                throughput_log.append((time.time() - wall_start, pbar1.n))

        if remaining_s2:
            done2, remaining_s2 = ray.wait(remaining_s2, num_returns=1, timeout=0.1)
            for f in done2:
                r2 = ray.get(f)
                s2_times.append(r2["s2_time"])
                creator = s2_futures[f]
                creator_texts[creator].append(r2["text"])
                pbar2.update(1)
                upd(s2_done=pbar2.n)
                if len(creator_texts[creator]) == expected[creator]:
                    combined = " ".join(creator_texts[creator])
                    s3f = clip.embed.remote(creator, creator_frames[creator], combined)
                    s3_futures[s3f] = creator

        if s3_futures:
            done3 = [f for f in list(s3_futures.keys())
                     if ray.wait([f], num_returns=1, timeout=0)[0]]
            for f in done3:
                r3 = ray.get(f)
                s3_times.append(r3["s3_time"])
                del s3_futures[f]
                pbar3.update(1)
                upd(s3_done=pbar3.n)

    pbar1.close(); pbar2.close(); pbar3.close()
    shutil.rmtree(bench_dir, ignore_errors=True)

    total_wall = round(time.time() - wall_start, 2)
    return {
        "total_wall_s":  total_wall,
        "n_videos":      n_videos,
        "n_creators":    len(unique_creators),
        "s1_avg":        round(np.mean(s1_times), 2),
        "s1_total":      round(sum(s1_times), 1),
        "s2_avg":        round(np.mean(s2_times), 2),
        "s2_total":      round(sum(s2_times), 1),
        "s3_avg":        round(np.mean(s3_times), 2),
        "s3_total":      round(sum(s3_times), 1),
        "throughput_vps": round(n_videos / total_wall, 2),
        "throughput_log": throughput_log,
    }


def plot_chart(r: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
        "axes.edgecolor":   "#30363d", "axes.labelcolor": "#e6edf3",
        "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
        "grid.color":       "#21262d", "text.color":      "#e6edf3",
        "font.family":      "monospace",
    })

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        f"TikTok Pipeline at Scale — {r['n_videos']} Videos  |  "
        f"{NUM_WORKERS} CPU Workers + 1 GPU\n"
        "Stage1=FFmpeg/CPU  |  Stage2=Whisper/GPU  |  Stage3=CLIP/GPU  |  All stages stream concurrently",
        fontsize=12, fontweight="bold", color="#e6edf3", y=1.02
    )

    # ── Panel 1: Throughput over time ─────────────────────────────────────────
    ax = axes[0]
    log = r["throughput_log"]
    if log:
        times   = [t for t, _ in log]
        done    = [d for _, d in log]
        # Smooth into 30 buckets
        bucket_t = np.linspace(0, times[-1], 30)
        bucket_d = np.interp(bucket_t, times, done)
        ax.fill_between(bucket_t / 60, bucket_d, alpha=0.3, color="#58a6ff")
        ax.plot(bucket_t / 60, bucket_d, color="#58a6ff", linewidth=2)
    ax.set_title("Videos Processed Over Time", fontsize=11, pad=10)
    ax.set_xlabel("Wall time (minutes)")
    ax.set_ylabel("Videos completed (Stage 1)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.text(0.05, 0.92,
            f"Total: {r['n_videos']} videos\n"
            f"{r['throughput_vps']} vid/s\n"
            f"Wall: {round(r['total_wall_s']/60, 1)} min",
            transform=ax.transAxes, fontsize=9,
            color="#3fb950", va="top",
            bbox=dict(boxstyle="round", facecolor="#21262d", alpha=0.8))

    # ── Panel 2: Stage breakdown bar ─────────────────────────────────────────
    ax = axes[1]
    stages  = ["Stage 1\nFFmpeg/CPU", "Stage 2\nWhisper/GPU", "Stage 3\nCLIP/GPU"]
    avgs    = [r["s1_avg"], r["s2_avg"], r["s3_avg"]]
    totals  = [r["s1_total"], r["s2_total"], r["s3_total"]]
    colors  = ["#58a6ff", "#bc8cff", "#3fb950"]
    bars    = ax.bar(stages, avgs, color=colors, alpha=0.85, width=0.5)
    for bar, avg, tot in zip(bars, avgs, totals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"avg {avg}s\ntotal {tot}s",
                ha="center", va="bottom", fontsize=9, color="#e6edf3")
    ax.set_title("Avg Time Per Video/Creator\n(stages run concurrently)", fontsize=11, pad=10)
    ax.set_ylabel("Average time (seconds)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Panel 3: Scale projection ─────────────────────────────────────────────
    ax = axes[2]
    scale_counts = [100, 250, 500, 762, 1000, 2500, 5000, 10000]
    ref_time_min = r["total_wall_s"] / 60
    ref_videos   = r["n_videos"]
    proj_8   = [ref_time_min * (n / ref_videos) for n in scale_counts]
    proj_16  = [t / 1.8 for t in proj_8]
    proj_32  = [t / 3.2 for t in proj_8]
    ax.plot(scale_counts, proj_8,  marker="o", linewidth=2,
            color="#58a6ff", label="8 CPU workers (current)")
    ax.plot(scale_counts, proj_16, marker="s", linewidth=2,
            color="#3fb950", linestyle="--", label="16 workers (2 servers)")
    ax.plot(scale_counts, proj_32, marker="^", linewidth=2,
            color="#f78166", linestyle="--", label="32 workers (4 servers)")
    # Mark current run
    ax.axvline(x=ref_videos, color="#8b949e", linestyle=":", linewidth=1)
    ax.text(ref_videos + 100, max(proj_8)*0.85,
            f"Current\n({ref_videos} videos)", fontsize=8, color="#8b949e")
    ax.set_title("Scale Projection\n(linear estimate)", fontsize=11, pad=10)
    ax.set_xlabel("Number of videos")
    ax.set_ylabel("Estimated time (minutes)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xscale("log")

    fig.text(0.5, -0.04,
             f"Hardware: {os.cpu_count()} CPU cores  |  1x GPU (CUDA, Whisper+CLIP shared 0.5+0.5)  |  "
             f"Total wall time: {round(r['total_wall_s']/60, 1)} min  |  "
             f"Throughput: {r['throughput_vps']} videos/s",
             ha="center", fontsize=9, color="#8b949e")

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Chart saved: {OUT_CHART}")


def _count_videos() -> int:
    total = 0
    for c in os.listdir(RESULTS_DIR):
        cdir = os.path.join(RESULTS_DIR, c)
        if os.path.isdir(cdir):
            total += len(glob.glob(os.path.join(cdir, "*.mp4")))
    return total


def _download_to_target(target: int = 1000):
    """Download new creators until we reach `target` total videos."""
    existing = _count_videos()
    if existing >= target:
        print(f"  Already have {existing} videos — skipping download.\n")
        return

    already_done    = set(os.listdir(RESULTS_DIR))
    videos_needed   = target - existing
    creators_needed = -(-videos_needed // MAX_VIDEOS_PER_CREATOR)  # ceiling

    print(f"  Videos on disk   : {existing}")
    print(f"  Target           : {target}")
    print(f"  Need ~{creators_needed} more creators ({videos_needed} videos)\n")

    df = pd.read_csv(CSV_PATH)
    df = df[df["video_count"] > 10]
    df = df[~df["username"].isin(already_done)]
    df = df.head(creators_needed)

    if df.empty:
        print("  No new creators found in CSV — proceeding with existing videos.\n")
        return

    print(f"  Downloading {len(df)} new creators...\n")
    tasks = [process_creator.remote(row._asdict()) for row in df.itertuples(index=False)]
    for i, fut in enumerate(tasks, 1):
        username = ray.get(fut)
        print(f"  [{i}/{len(tasks)}] downloaded: {username}")

    print(f"\n  Download complete. Total videos now: {_count_videos()}\n")


def main():
    start_web()
    ray.init(
        include_dashboard=True, dashboard_host="0.0.0.0",
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
        num_gpus=1 if torch.cuda.is_available() else 0,
    )
    print(f"\n  Ray ready  |  {ray.cluster_resources()}")
    print(f"  Ray Dashboard  http://localhost:8265")
    print(f"  Live progress  http://localhost:8888\n")

    # ── Step 0: top up to 1000 videos if needed ───────────────────────────────
    print("=" * 55)
    print("  STEP 0 — Checking video count / downloading if needed")
    print("=" * 55)
    _download_to_target(target=1000)

    # Collect all mp4 files
    all_mp4s = []
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir): continue
        all_mp4s.extend(glob.glob(os.path.join(cdir, "*.mp4"))[:3])

    print(f"  Videos available : {len(all_mp4s)}")
    print(f"  Workers          : {NUM_WORKERS} CPU + 1 GPU\n")

    print("  Loading GPU actors (Whisper + CLIP on CUDA)...")
    whisper = GPUWhisperActor.remote()
    clip    = GPUCLIPActor.remote()
    time.sleep(5)   # let actors initialise

    results = run_large(all_mp4s, whisper, clip)

    # Summary
    print(f"\n{'='*55}")
    print(f"  RESULTS — {results['n_videos']} videos  |  {NUM_WORKERS} CPU + 1 GPU")
    print(f"{'='*55}")
    print(f"  Total wall time   : {results['total_wall_s']}s  "
          f"({round(results['total_wall_s']/60, 1)} min)")
    print(f"  Throughput        : {results['throughput_vps']} videos/s")
    print(f"  Stage 1 avg/video : {results['s1_avg']}s  (FFmpeg CPU)")
    print(f"  Stage 2 avg/video : {results['s2_avg']}s  (Whisper GPU)")
    print(f"  Stage 3 avg/creator: {results['s3_avg']}s (CLIP GPU)")
    print(f"  Projection 10k    : ~{round(10000/results['throughput_vps']/60, 0):.0f} min "
          f"with same hardware")
    print(f"{'='*55}")

    # Save CSV
    pd.DataFrame([{k: v for k, v in results.items() if k != "throughput_log"}]).to_csv(
        OUT_CSV, index=False
    )
    print(f"  Saved: {OUT_CSV}")

    plot_chart(results)
    upd(finished=True)

    print("\n  Dashboard still live at http://localhost:8888 — Ctrl+C to exit")
    ray.shutdown()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
