"""
benchmark_1000.py  —  Video Scalability Benchmark (single machine)
==================================================================
Same architecture as full_pipeline_benchmark.py:

  Stage 1  FFmpeg   — CPU, per-video extract to disk (wav + frames)
  Stage 2  Whisper  — GPU 2×0.25, round-robin, parallel transcription
  Stage 3  CLIP     — GPU 0.5, fires per-video when S2 finishes

Worker configs : 8 / 12 / 16 CPU workers (S1 parallelism cap)

Run:
  python benchmark_1000.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob, time, subprocess, shutil, threading
import numpy as np, pandas as pd, ray, psutil, torch
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR     = "results_4"
TARGET_VIDEOS   = 200
PHYSICAL_CORES  = 8
LOGICAL_CPUS    = 16
CPU_CONFIGS     = [8, 12, 16]   # max concurrent S1 FFmpeg tasks
WHISPER_ACTORS  = 2          # GPU whisper pool (2 × 0.25 GPU + CLIP 0.5 = 1.0)
AUDIO_MAX_SEC   = None          # max audio seconds for Whisper; None = full-length
OUT_CSV         = "benchmark_1k_results.csv"
OUT_CHART       = "benchmark_1k_chart.png"
RAY_ADDRESS     = None


# ── Stage 1: FFmpeg per video (CPU) ───────────────────────────────────────────
@ray.remote(num_cpus=1)
def stage1_extract(mp4_path: str, out_dir: str) -> dict:
    """Extract audio WAV + JPEG frames to disk — same pattern as full_pipeline."""
    import tempfile, shutil as _shutil
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    base       = os.path.splitext(os.path.basename(mp4_path))[0]
    wav_path   = os.path.join(out_dir, f"{base}.wav")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    audio_args = ["-map", "0:a", "-vn",
                  "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
    if AUDIO_MAX_SEC is not None:
        audio_args = ["-t", str(AUDIO_MAX_SEC)] + audio_args

    subprocess.run(
        ["ffmpeg", "-threads", "1", "-y", "-i", mp4_path]
        + audio_args + [wav_path,
         "-ss", "1", "-t", "10", "-map", "0:v",
         "-vf", "fps=1,scale=224:224", "-qscale:v", "3",
         os.path.join(frames_dir, f"{base}_%02d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:10]
    return {
        "mp4":     mp4_path,
        "wav":     wav_path if os.path.exists(wav_path) else "",
        "frames":  frames,
        "s1_time": round(time.time() - t0, 3),
        "creator": os.path.basename(os.path.dirname(mp4_path)),
    }


# ── Stage 2: Whisper on GPU (2×0.25 — parallel instances, shares GPU with CLIP) ─
@ray.remote(num_gpus=0.25, num_cpus=0)
class GPUWhisperActor:
    def __init__(self):
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctype  = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel("tiny", device=device, compute_type=ctype)
        print(f"  [Whisper] loaded on {device.upper()}")

    def transcribe(self, wav_path: str) -> dict:
        t0 = time.time()
        try:
            segs, _ = self.model.transcribe(
                wav_path, beam_size=1, vad_filter=False, without_timestamps=True
            )
            text = " ".join(s.text.strip() for s in segs)
        except Exception:
            text = ""
        return {"text": text, "s2_time": round(time.time() - t0, 3)}


# ── Stage 3: CLIP on GPU (0.5 — shares with Whisper) ─────────────────────────
@ray.remote(num_gpus=0.5, num_cpus=0, max_concurrency=4)
class GPUCLIPActor:
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
        print(f"  [CLIP] loaded on {self.device.upper()}")

    def embed(self, frame_paths: list, text: str) -> dict:
        from PIL import Image
        t0 = time.time()
        tensors = []
        for fp in (frame_paths or [])[:10]:
            try:
                tensors.append(self.preprocess(Image.open(fp).convert("RGB")))
            except Exception:
                pass
        if tensors:
            with torch.no_grad():
                batch = torch.stack(tensors).to(self.device)
                if self.device == "cuda":
                    batch = batch.half()
                self.model.encode_image(batch)
        if text and text.strip():
            with torch.no_grad():
                self.model.encode_text(
                    self.tokenizer([text[:77]]).to(self.device))
        return {"s3_time": round(time.time() - t0, 3)}


# ── Resource monitor ───────────────────────────────────────────────────────────
_mon_active = False
_mon_log: list = []

def _monitor_loop():
    while _mon_active:
        cpu = psutil.cpu_percent(interval=1)
        gpu = 0.0
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu = round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e6, 1)
            except Exception:
                gpu = round(torch.cuda.memory_allocated() / 1e6, 1)
        _mon_log.append({"cpu_pct": cpu, "gpu_mb": gpu})
        time.sleep(1)

def start_monitor():
    global _mon_active, _mon_log
    _mon_active, _mon_log = True, []
    threading.Thread(target=_monitor_loop, daemon=True).start()

def stop_monitor():
    global _mon_active
    _mon_active = False
    time.sleep(0.3)


# ── Streaming pipeline (same pattern as full_pipeline_benchmark.py) ────────────
def run_pipeline(num_workers: int, mp4_list: list,
                 whisper_pool: list, clip: GPUCLIPActor,
                 bench_dir: str) -> dict:
    """
    Per-video streaming pipeline:
      S1: up to num_workers FFmpeg tasks in flight
      S2: round-robin across whisper_pool (2× GPU actors in parallel)
      S3: GPU CLIP fires per video when S2 done
    """
    n_videos = len(mp4_list)
    n_creators = len(set(os.path.basename(os.path.dirname(p)) for p in mp4_list))
    os.makedirs(bench_dir, exist_ok=True)

    s1_queue   = list(mp4_list)
    s1_futures = {}          # future → mp4
    s1_times, s2_times, s3_times = [], [], []
    s2_futures = {}          # future → (frames, mp4)
    s3_futures = {}          # future → mp4
    _w_idx = [0]

    def _submit_s1():
        while s1_queue and len(s1_futures) < num_workers:
            mp4 = s1_queue.pop(0)
            creator  = os.path.basename(os.path.dirname(mp4))
            vid_name = os.path.splitext(os.path.basename(mp4))[0]
            out_dir  = os.path.join(bench_dir, creator, vid_name)
            fut = stage1_extract.remote(mp4, out_dir)
            s1_futures[fut] = mp4

    _submit_s1()

    pbar1 = tqdm(total=n_videos, desc=f"S1-FFmpeg ({num_workers:2d}w)", ncols=72,
                 colour="blue",    position=0, leave=True)
    pbar2 = tqdm(total=n_videos, desc=f"S2-Whisper({WHISPER_ACTORS}×GPU)", ncols=72,
                 colour="magenta", position=1, leave=True)
    pbar3 = tqdm(total=n_videos, desc=f"S3-CLIP   GPU   ", ncols=72,
                 colour="green",   position=2, leave=True)

    wall_start = time.time()

    while s1_futures or s1_queue or s2_futures or s3_futures:
        # ── Poll S1
        if s1_futures:
            done1, _ = ray.wait(list(s1_futures), num_returns=1, timeout=0.05)
            for f in done1:
                r1 = ray.get(f)
                mp4 = s1_futures.pop(f)
                s1_times.append(r1["s1_time"])
                pbar1.update(1)
                if r1["wav"]:
                    w = whisper_pool[_w_idx[0] % len(whisper_pool)]
                    _w_idx[0] += 1
                    s2f = w.transcribe.remote(r1["wav"])
                    s2_futures[s2f] = (r1["frames"], mp4)
            _submit_s1()

        # ── Poll S2
        if s2_futures:
            done2, _ = ray.wait(list(s2_futures), num_returns=min(4, len(s2_futures)),
                                timeout=0.05)
            for f in done2:
                r2 = ray.get(f)
                frames, mp4 = s2_futures.pop(f)
                s2_times.append(r2["s2_time"])
                pbar2.update(1)
                s3f = clip.embed.remote(frames, r2["text"])
                s3_futures[s3f] = mp4

        # ── Poll S3
        if s3_futures:
            done3, _ = ray.wait(list(s3_futures), num_returns=min(4, len(s3_futures)),
                                timeout=0.05)
            for f in done3:
                r3 = ray.get(f)
                s3_times.append(r3["s3_time"])
                s3_futures.pop(f)
                pbar3.update(1)

    pbar1.close(); pbar2.close(); pbar3.close()
    shutil.rmtree(bench_dir, ignore_errors=True)

    total_wall = round(time.time() - wall_start, 2)
    return {
        "cpu_workers":    num_workers,
        "n_videos":       n_videos,
        "n_creators":     n_creators,
        "s1_avg_s":       round(np.mean(s1_times), 3) if s1_times else 0,
        "s2_avg_s":       round(np.mean(s2_times), 3) if s2_times else 0,
        "s3_avg_s":       round(np.mean(s3_times), 3) if s3_times else 0,
        "total_wall_s":   total_wall,
        "throughput_vps": round(n_videos / total_wall, 3) if total_wall else 0,
        "avg_cpu_pct":    round(np.mean([m["cpu_pct"] for m in _mon_log]), 1) if _mon_log else 0,
        "peak_gpu_mb":    round(max((m["gpu_mb"] for m in _mon_log), default=0), 1),
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

    workers  = [r["cpu_workers"]    for r in results]
    walls    = [r["total_wall_s"]   for r in results]
    tputs    = [r["throughput_vps"] for r in results]
    cpus     = [r["avg_cpu_pct"]    for r in results]
    base_w   = walls[0]
    base_workers = workers[0]
    speedups = [round(base_w / w, 2) for w in walls]
    ideal    = [round(w / base_workers, 2) for w in workers]
    cmap     = plt.cm.Blues(np.linspace(0.45, 0.95, len(workers)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Video Benchmark — single machine ({PHYSICAL_CORES}c/{LOGICAL_CPUS}t)\n"
        f"S1 workers: {' / '.join(str(w) for w in workers)}  |  "
        f"S2={WHISPER_ACTORS}×Whisper@0.25  S3=CLIP@0.5  |  "
        "S1=FFmpeg/CPU  S2=Whisper/GPU  S3=CLIP/GPU",
        fontsize=11, fontweight="bold", color="#e6edf3", y=1.02
    )

    ax = axes[0]
    bars = ax.bar([str(w) for w in workers], walls, color=cmap, alpha=0.9, width=0.55)
    for bar, val in zip(bars, walls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}s", ha="center", va="bottom", fontsize=9)
    ax.set_title("Total Wall Time (seconds)", fontsize=11, pad=8)
    ax.set_xlabel("S1 CPU Workers"); ax.set_ylabel("Wall time (s)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]; ax2 = ax.twinx()
    ax.bar([str(w) for w in workers], tputs, color=cmap, alpha=0.75, width=0.55)
    ax2.plot([str(w) for w in workers], speedups,
             color="#f0e68c", marker="o", linewidth=2, label="Measured")
    ax2.plot([str(w) for w in workers], ideal,
             color="#8b949e", linestyle="--", marker="x", linewidth=1.5, label="Ideal linear")
    for i, sp in enumerate(speedups):
        ax2.text(i, sp + 0.05, f"×{sp}", ha="center", fontsize=8, color="#f0e68c")
    ax.set_title("Throughput & Speedup", fontsize=11, pad=8)
    ax.set_xlabel("S1 CPU Workers"); ax.set_ylabel("Videos / second")
    ax2.set_ylabel(f"Speedup vs {base_workers} workers")
    ax2.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[2]
    bars = ax.bar([str(w) for w in workers], cpus, color=cmap, alpha=0.9, width=0.55)
    for bar, val in zip(bars, cpus):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Average CPU Utilisation", fontsize=11, pad=8)
    ax.set_xlabel("S1 CPU Workers"); ax.set_ylabel("CPU %")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Chart saved: {OUT_CHART}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    try:
        ray.shutdown()
    except Exception:
        pass

    n_cpus = min(LOGICAL_CPUS, os.cpu_count() or LOGICAL_CPUS)
    configs = [w for w in CPU_CONFIGS if w <= n_cpus]
    if not configs:
        configs = [n_cpus]

    if RAY_ADDRESS:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    else:
        ray.init(address="local",
                 num_cpus=n_cpus,
                 num_gpus=1 if torch.cuda.is_available() else 0,
                 ignore_reinit_error=True,
                 runtime_env={"env_vars": {"OMP_NUM_THREADS": "1",
                                           "MKL_NUM_THREADS": "1"}})

    print(f"\n  Single-machine benchmark (full_pipeline architecture)")
    print(f"  Hardware     : {PHYSICAL_CORES} cores / {n_cpus} logical CPUs")
    print(f"  S1 worker runs : {configs}")
    print(f"  Ray ready      |  {ray.cluster_resources()}")
    print(f"  Dashboard      http://localhost:8265\n")

    real_mp4s = []
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir) or creator.startswith("_"):
            continue
        real_mp4s.extend(sorted(glob.glob(os.path.join(cdir, "*.mp4"))))

    if not real_mp4s:
        print("  ERROR: No .mp4 files found in results_4/.")
        ray.shutdown(); return

    all_mp4s = (real_mp4s * ((TARGET_VIDEOS // len(real_mp4s)) + 1))[:TARGET_VIDEOS]
    unique_creators = list(dict.fromkeys(
        os.path.basename(os.path.dirname(p)) for p in real_mp4s
    ))
    audio_label = f"{AUDIO_MAX_SEC}s max" if AUDIO_MAX_SEC is not None else "full-length"
    print(f"  Real videos : {len(real_mp4s)}  ({len(unique_creators)} creators)")
    print(f"  Tasks       : {len(all_mp4s)} (cycled to reach {TARGET_VIDEOS})")
    print(f"  Audio trim  : {audio_label}\n")

    print(f"  Loading GPU actors ({WHISPER_ACTORS}×Whisper@0.25 + CLIP@0.5)...")
    whisper_pool = [GPUWhisperActor.remote() for _ in range(WHISPER_ACTORS)]
    clip_actor   = GPUCLIPActor.remote()
    try:
        ray.get(clip_actor.embed.remote([], ""), timeout=60)
    except Exception:
        pass
    try:
        ray.get([w.transcribe.remote("") for w in whisper_pool], timeout=120)
    except Exception:
        pass
    print("  Models ready.\n")

    bench_dir = os.path.join(RESULTS_DIR, "_bench_tmp_1k")
    results, baseline_wall = [], None

    for n_workers in configs:
        print(f"\n{'='*60}")
        print(f"  RUN: {n_workers} S1 CPU workers  |  {len(all_mp4s)} videos")
        print(f"{'='*60}")

        start_monitor()
        try:
            row = run_pipeline(n_workers, all_mp4s, whisper_pool, clip_actor, bench_dir)
        except Exception as e:
            stop_monitor()
            print(f"\n  [SKIP] {n_workers}w run failed: {e}")
            shutil.rmtree(bench_dir, ignore_errors=True)
            continue
        stop_monitor()

        if baseline_wall is None:
            baseline_wall = row["total_wall_s"]
        row["speedup"] = round(baseline_wall / row["total_wall_s"], 2)
        results.append(row)

        print(f"\n  Wall {row['total_wall_s']}s  |  "
              f"{row['throughput_vps']} vid/s  |  "
              f"S1 {row['s1_avg_s']}s  S2 {row['s2_avg_s']}s  S3 {row['s3_avg_s']}s  |  "
              f"Speedup x{row['speedup']}")

    if not results:
        print("  No runs completed.")
        ray.shutdown(); return

    cols = ["cpu_workers", "n_videos", "n_creators",
            "s1_avg_s", "s2_avg_s", "s3_avg_s",
            "total_wall_s", "throughput_vps", "speedup",
            "avg_cpu_pct", "peak_gpu_mb"]
    df = pd.DataFrame(results)

    print(f"\n\n{'='*75}")
    print(f"  FINAL RESULTS")
    print(f"{'='*75}")
    print(df[cols].to_string(index=False))
    print(f"{'='*75}")

    df[cols].to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")
    plot_chart(results)
    ray.shutdown()


if __name__ == "__main__":
    main()
