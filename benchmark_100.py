"""
benchmark_100.py  —  100-Video Scalability Benchmark
=====================================================
Runs the full concurrent pipeline on exactly 100 videos
(up to 34 creators × 3 videos each) across three CPU
worker configurations:  8 → 16 → 32 CPUs  +  1 GPU.

Stages (all overlap / stream concurrently):
  Stage 1  FFmpeg   — CPU  — extract audio + frames
  Stage 2  Whisper  — CPU  — transcribe WAV
  Stage 3  CLIP     — GPU  — multimodal embedding

Output:
  benchmark_100_results.csv   (one row per worker config)
  benchmark_100_chart.png     (3-panel chart)
  stats/pipeline_stats.txt    (full dataset stats, auto-updated)

Single-machine run:
  python benchmark_100.py

Two-laptop cluster run:
  Laptop A (head, GPU):   ray start --head --port=6379 --dashboard-host=0.0.0.0
  Laptop B (worker CPUs): ray start --address="<LAPTOP_A_IP>:6379"
  Then change RAY_ADDRESS below from None to "auto".
"""

import os, glob, time, threading, collections
import numpy as np, pandas as pd, ray, psutil, torch
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR  = "results_4"
TARGET_VIDEOS = 100          # total videos to process (3 per creator → ~34 creators)
CPU_CONFIGS   = [8, 16, 32]  # worker counts to test
OUT_CSV       = "benchmark_100_results.csv"
OUT_CHART     = "benchmark_100_chart.png"

# Set to "auto" when running a two-laptop Ray cluster.
# Leave as None for single-machine mode.
RAY_ADDRESS   = "auto"


# ── Stage 1: FFmpeg extract (CPU remote task) ──────────────────────────────────
@ray.remote(num_cpus=1)
def stage1_extract(mp4_path: str, out_dir: str) -> dict:
    """Extract audio WAV + frames PNG from one MP4."""
    import subprocess, shutil
    t0 = time.time()
    creator = os.path.basename(os.path.dirname(mp4_path))
    os.makedirs(out_dir, exist_ok=True)
    wav_path    = os.path.join(out_dir, "audio.wav")
    frames_dir  = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Audio
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-ac", "1", "-ar", "16000",
         "-f", "wav", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    # Frames (1 fps, first 10 s)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-t", "10", "-vf", "fps=1",
         os.path.join(frames_dir, "frame_%03d.png")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    frames = glob.glob(os.path.join(frames_dir, "*.png"))
    return {
        "creator": creator,
        "wav":     wav_path,
        "frames":  frames,
        "s1_time": round(time.time() - t0, 3),
    }


# ── Stage 2: Whisper CPU remote function (lazy global model per worker process)
# Ray reuses worker processes, so the model loads once per process — not once
# per task. This means 32 concurrent tasks share at most 32 worker processes,
# each loading the model only when first used. No upfront OOM spike.
_whisper_model = None

@ray.remote(num_cpus=1)
def transcribe_task(wav_path: str) -> dict:
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    t0 = time.time()
    try:
        segs, _ = _whisper_model.transcribe(
            wav_path, beam_size=1, vad_filter=False, without_timestamps=True
        )
        text = " ".join(s.text.strip() for s in segs)
    except Exception:
        text = ""
    return {
        "wav":     wav_path,
        "text":    text,
        "s2_time": round(time.time() - t0, 3),
    }


# ── Stage 3: CLIP GPU actor ────────────────────────────────────────────────────
@ray.remote(num_gpus=1)
class GPUEmbeddingActor:
    def __init__(self):
        import open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()
        if self.device == "cuda":
            self.model.half()

    def embed(self, creator: str, frame_paths: list, transcription: str) -> dict:
        from PIL import Image
        t0 = time.time()
        tensors = []
        for f in frame_paths[:10]:
            try:
                tensors.append(self.preprocess(Image.open(f).convert("RGB")))
            except Exception:
                pass
        if tensors:
            with torch.no_grad():
                batch = torch.stack(tensors).to(self.device)
                if self.device == "cuda":
                    batch = batch.half()
                self.model.encode_image(batch)
        if transcription.strip():
            with torch.no_grad():
                self.model.encode_text(
                    self.tokenizer([transcription[:77]]).to(self.device)
                )
        return {"creator": creator, "s3_time": round(time.time() - t0, 3)}


# ── Live resource monitor ──────────────────────────────────────────────────────
_mon_active = False
_mon_log: list = []

def _monitor_thread():
    while _mon_active:
        snap = {
            "cpu_pct": psutil.cpu_percent(interval=1),
            "ram_gb":  round(psutil.virtual_memory().used / 1e9, 2),
            "gpu_mb":  round(torch.cuda.memory_allocated() / 1e6, 1)
                       if torch.cuda.is_available() else 0,
        }
        _mon_log.append(snap)
        time.sleep(2)

def start_monitor():
    global _mon_active, _mon_log
    _mon_active, _mon_log = True, []
    threading.Thread(target=_monitor_thread, daemon=True).start()

def stop_monitor():
    global _mon_active
    _mon_active = False
    time.sleep(0.3)


# ── One full concurrent pipeline run ─────────────────────────────────────────
def run_pipeline(num_cpu_workers: int, mp4_list: list,
                 gpu_actor: GPUEmbeddingActor) -> dict:
    """
    Streaming concurrent pipeline:
      S1 tasks → as each mp4 finishes → S2 task fires immediately
      S2 tasks → as soon as a creator's last wav finishes → S3 fires
    """
    creator_of    = {p: os.path.basename(os.path.dirname(p)) for p in mp4_list}
    unique_creators = list(dict.fromkeys(creator_of.values()))
    expected      = collections.Counter(creator_of.values())

    bench_dir = os.path.join(RESULTS_DIR, "_bench_tmp_100")
    os.makedirs(bench_dir, exist_ok=True)

    # No actor pool needed — transcribe_task is a stateless remote function.
    # Ray spawns up to num_cpu_workers worker processes on demand across the
    # cluster, each loading the Whisper model lazily on first use.

    # Submit all Stage-1 tasks
    s1_futures: dict = {}
    for mp4 in mp4_list:
        creator = creator_of[mp4]
        vid_name = os.path.splitext(os.path.basename(mp4))[0]
        out_dir  = os.path.join(bench_dir, creator, vid_name)
        fut = stage1_extract.remote(mp4, out_dir)
        s1_futures[fut] = mp4

    creator_frames: dict = {c: [] for c in unique_creators}
    creator_texts:  dict = {c: [] for c in unique_creators}
    s1_times, s2_times, s3_times = [], [], []
    s2_futures: dict = {}   # fut -> creator
    s3_futures: dict = {}   # fut -> creator
    worker_idx = 0

    pbar1 = tqdm(total=len(mp4_list), desc=f"S1-FFmpeg ({num_cpu_workers:2d}w)", ncols=72, colour="blue")
    pbar2 = tqdm(total=len(mp4_list), desc=f"S2-Whisper({num_cpu_workers:2d}w)", ncols=72, colour="magenta")
    pbar3 = tqdm(total=len(unique_creators), desc=f"S3-CLIP   ({num_cpu_workers:2d}w)", ncols=72, colour="green")

    wall_start   = time.time()
    remaining_s1 = list(s1_futures.keys())
    remaining_s2: list = []

    while remaining_s1 or remaining_s2 or s3_futures:
        # Poll Stage 1
        if remaining_s1:
            done1, remaining_s1 = ray.wait(remaining_s1, num_returns=1, timeout=0.05)
            for f in done1:
                r1 = ray.get(f)
                s1_times.append(r1["s1_time"])
                creator = r1["creator"]
                creator_frames[creator].extend(r1["frames"])
                # Fire Stage 2 immediately
                worker_idx += 1
                s2f = transcribe_task.remote(r1["wav"])
                s2_futures[s2f] = creator
                remaining_s2.append(s2f)
                pbar1.update(1)

        # Poll Stage 2
        if remaining_s2:
            done2, remaining_s2 = ray.wait(remaining_s2, num_returns=1, timeout=0.05)
            for f in done2:
                r2 = ray.get(f)
                s2_times.append(r2["s2_time"])
                creator = s2_futures[f]
                creator_texts[creator].append(r2["text"])
                pbar2.update(1)
                # Fire Stage 3 when all videos for this creator are done
                if len(creator_texts[creator]) == expected[creator]:
                    combined = " ".join(creator_texts[creator])
                    s3f = gpu_actor.embed.remote(
                        creator, creator_frames[creator], combined
                    )
                    s3_futures[s3f] = creator

        # Poll Stage 3
        if s3_futures:
            done3 = [f for f in list(s3_futures)
                     if ray.wait([f], num_returns=1, timeout=0)[0]]
            for f in done3:
                r3 = ray.get(f)
                s3_times.append(r3["s3_time"])
                del s3_futures[f]
                pbar3.update(1)

    pbar1.close(); pbar2.close(); pbar3.close()

    import shutil
    shutil.rmtree(bench_dir, ignore_errors=True)

    total_wall = round(time.time() - wall_start, 2)
    avg_cpu = round(np.mean([m["cpu_pct"] for m in _mon_log]), 1) if _mon_log else 0
    peak_gpu = round(max((m["gpu_mb"] for m in _mon_log), default=0), 1)

    return {
        "cpu_workers":     num_cpu_workers,
        "n_creators":      len(unique_creators),
        "n_videos":        len(mp4_list),
        "s1_avg_s":        round(np.mean(s1_times), 2) if s1_times else 0,
        "s2_avg_s":        round(np.mean(s2_times), 2) if s2_times else 0,
        "s3_avg_s":        round(np.mean(s3_times), 2) if s3_times else 0,
        "total_wall_s":    total_wall,
        "throughput_vps":  round(len(mp4_list) / total_wall, 3),
        "avg_cpu_pct":     avg_cpu,
        "peak_gpu_mb":     peak_gpu,
    }


# ── Chart ──────────────────────────────────────────────────────────────────────
def plot_chart(results: list):
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

    workers  = [r["cpu_workers"]   for r in results]
    walls    = [r["total_wall_s"]  for r in results]
    tputs    = [r["throughput_vps"] for r in results]
    cpus     = [r["avg_cpu_pct"]   for r in results]
    speedups = [round(walls[0] / w, 2) for w in walls]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"100-Video Benchmark  —  8 / 16 / 32 CPU Workers + 1 GPU\n"
        "Stage1=FFmpeg/CPU  |  Stage2=Whisper/CPU  |  Stage3=CLIP/GPU  |  Streaming concurrent",
        fontsize=11, fontweight="bold", color="#e6edf3", y=1.02
    )
    colors = ["#58a6ff", "#3fb950", "#f78166"]

    # Panel 1: Wall time
    ax = axes[0]
    bars = ax.bar([str(w) for w in workers], walls, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, walls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}s", ha="center", va="bottom", fontsize=10)
    ax.set_title("Total Wall Time (seconds)", fontsize=11, pad=8)
    ax.set_xlabel("CPU Workers")
    ax.set_ylabel("Wall time (s)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Throughput + speedup
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar([str(w) for w in workers], tputs, color=colors, alpha=0.75, width=0.5, label="Throughput")
    ax2.plot([str(w) for w in workers], speedups, color="#f0e68c",
             marker="o", linewidth=2, label="Speedup vs 8w")
    for i, (w, sp) in enumerate(zip(workers, speedups)):
        ax2.text(i, sp + 0.05, f"×{sp}", ha="center", fontsize=9, color="#f0e68c")
    ax.set_title("Throughput & Speedup", fontsize=11, pad=8)
    ax.set_xlabel("CPU Workers")
    ax.set_ylabel("Videos / second")
    ax2.set_ylabel("Speedup vs 8 workers")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Panel 3: CPU utilisation
    ax = axes[2]
    ax.bar([str(w) for w in workers], cpus, color=colors, alpha=0.85, width=0.5)
    for i, (bar, val) in enumerate(zip(ax.patches, cpus)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Average CPU Utilisation", fontsize=11, pad=8)
    ax.set_xlabel("CPU Workers")
    ax.set_ylabel("CPU %")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Chart saved: {OUT_CHART}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Ray init ──────────────────────────────────────────────────────────
    if RAY_ADDRESS:
        # Two-laptop cluster mode
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    else:
        # Single-machine mode — use all available CPUs + GPU
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=1 if torch.cuda.is_available() else 0,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            ignore_reinit_error=True,
        )
    print(f"\n  Ray ready  |  {ray.cluster_resources()}")
    print(f"  Dashboard  http://localhost:8265\n")

    # ── Collect all available videos, then repeat to reach TARGET_VIDEOS ─────
    real_mp4s = []
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir) or creator.startswith("_"):
            continue
        vids = sorted(glob.glob(os.path.join(cdir, "*.mp4")))[:3]
        if len(vids) == 3:
            real_mp4s.extend(vids)

    if not real_mp4s:
        print("  ERROR: No videos found in results_4/.")
        ray.shutdown()
        return

    # Repeat the real video list to reach TARGET_VIDEOS (benchmark throughput only)
    all_mp4s = (real_mp4s * ((TARGET_VIDEOS // len(real_mp4s)) + 1))[:TARGET_VIDEOS]
    unique = list(dict.fromkeys(os.path.basename(os.path.dirname(p)) for p in real_mp4s))
    print(f"  Real videos on disk : {len(real_mp4s)}  ({len(unique)} creators)")
    print(f"  Tasks to process    : {len(all_mp4s)}  (repeated to reach {TARGET_VIDEOS})")
    if len(real_mp4s) < 3:
        print("  ERROR: Not enough videos in results_4/. Run the download pipeline first.")
        ray.shutdown()
        return

    # ── Spin up ONE shared GPU actor for all runs ─────────────────────────
    print("  Loading CLIP GPU actor...")
    gpu_actor = GPUEmbeddingActor.remote()
    time.sleep(4)

    results = []
    baseline_wall = None

    for n_workers in CPU_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  RUN: {n_workers} CPU workers  |  {len(all_mp4s)} videos  |  {len(unique)} creators")
        print(f"{'='*60}")

        start_monitor()
        row = run_pipeline(n_workers, all_mp4s, gpu_actor)
        stop_monitor()

        if baseline_wall is None:
            baseline_wall = row["total_wall_s"]
        row["speedup"] = round(baseline_wall / row["total_wall_s"], 2)
        results.append(row)

        print(f"\n  Wall {row['total_wall_s']}s  |  "
              f"{row['throughput_vps']} vid/s  |  "
              f"Speedup x{row['speedup']}  |  "
              f"CPU avg {row['avg_cpu_pct']}%  |  "
              f"GPU peak {row['peak_gpu_mb']} MB")

    # ── Summary table ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    cols = ["cpu_workers", "n_videos", "n_creators",
            "s1_avg_s", "s2_avg_s", "s3_avg_s",
            "total_wall_s", "throughput_vps", "speedup",
            "avg_cpu_pct", "peak_gpu_mb"]

    print(f"\n\n{'='*75}")
    print(f"  FINAL RESULTS — 100-Video Benchmark  (8 / 16 / 32 CPU Workers + GPU)")
    print(f"{'='*75}")
    print(df[cols].to_string(index=False))
    print(f"{'='*75}")

    df[cols].to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")

    plot_chart(results)

    # ── Auto-update stats report ───────────────────────────────────────────
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "collect_stats", os.path.join("stats", "collect_stats.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        print("  Stats report updated: stats/pipeline_stats.txt")
    except Exception as e:
        print(f"  (Stats update skipped: {e})")

    ray.shutdown()


if __name__ == "__main__":
    main()
