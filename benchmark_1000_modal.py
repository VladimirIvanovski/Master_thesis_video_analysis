"""
benchmark_1000_modal.py — Local S1 + CLIP, Whisper on Modal T4 GPU
=================================================================
  Stage 1  FFmpeg   — local Ray CPU
  Stage 2  Whisper  — Modal 4×T4 containers, 2 parallel each (8 total)
  Stage 3  CLIP     — local Ray GPU (4090)

Setup (run once, do NOT put secrets in this file):
  pip install modal
  modal setup                          # official Modal login (recommended)
  # optional custom secret (never commit the value):
  modal secret create custom-secret vladimir2=<your-value>

Run:
  python benchmark_1000_modal.py

Budget tip (~$1): start with TARGET_VIDEOS=20, one CPU_CONFIGS entry.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob, time, subprocess, shutil, threading
import numpy as np, pandas as pd, ray, psutil, torch
from tqdm import tqdm

from modal_whisper_app import (
    app, ModalWhisper, MODAL_GPU, MODAL_MAX_CONTAINERS, MODAL_CONCURRENT,
)
MODAL_MAX_INFLIGHT = MODAL_MAX_CONTAINERS * MODAL_CONCURRENT  # 8 parallel S2 jobs

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR      = "results_4"
TARGET_VIDEOS    = 50            # raise after budget test (200 burns $ fast)
PHYSICAL_CORES   = 8
LOGICAL_CPUS     = 16
CPU_CONFIGS      = [8]           # benchmark label (S1 uses S1_WORKERS below)
S1_WORKERS       = LOGICAL_CPUS  # 16 parallel FFmpeg — S2 on Modal, no local CPU fight
FFMPEG_THREADS   = 1             # 1 thread/process × 16 workers = 16 CPUs
AUDIO_MAX_SEC    = None
OUT_CSV          = "benchmark_1k_modal_results.csv"
OUT_CHART        = "benchmark_1k_modal_chart.png"
RAY_ADDRESS      = None


# ── Local Stage 1: FFmpeg (CPU) ───────────────────────────────────────────────
@ray.remote(num_cpus=1)
def stage1_extract(mp4_path: str, out_dir: str) -> dict:
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
        ["ffmpeg", "-threads", str(FFMPEG_THREADS), "-y", "-i", mp4_path]
        + audio_args + [wav_path,
         "-ss", "1", "-t", "10", "-map", "0:v",
         "-vf", "fps=1,scale=224:224", "-qscale:v", "3",
         os.path.join(frames_dir, f"{base}_%02d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:10]
    return {
        "mp4": mp4_path,
        "wav": wav_path if os.path.exists(wav_path) else "",
        "frames": frames,
        "s1_time": round(time.time() - t0, 3),
        "creator": os.path.basename(os.path.dirname(mp4_path)),
    }


# ── Local Stage 3: CLIP (GPU) ─────────────────────────────────────────────────
@ray.remote(num_gpus=1, num_cpus=0, max_concurrency=4)
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


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(num_workers: int, mp4_list: list,
                 modal_whisper: ModalWhisper, clip: GPUCLIPActor,
                 bench_dir: str) -> dict:
    n_videos = len(mp4_list)
    n_creators = len(set(os.path.basename(os.path.dirname(p)) for p in mp4_list))
    os.makedirs(bench_dir, exist_ok=True)

    s1_queue   = list(mp4_list)
    s1_futures = {}
    s1_times, s2_times, s3_times = [], [], []
    s2_pending = []          # (wav_path, frames, mp4) waiting for Modal slot
    s2_calls   = {}          # Modal FunctionCall → (frames, mp4)
    s3_futures = {}

    def _submit_s1():
        while s1_queue and len(s1_futures) < S1_WORKERS:
            mp4 = s1_queue.pop(0)
            creator  = os.path.basename(os.path.dirname(mp4))
            vid_name = os.path.splitext(os.path.basename(mp4))[0]
            out_dir  = os.path.join(bench_dir, creator, vid_name)
            fut = stage1_extract.remote(mp4, out_dir)
            s1_futures[fut] = mp4

    def _submit_s2(wav_path: str, frames: list, mp4: str):
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        call = modal_whisper.transcribe.spawn(wav_bytes)
        s2_calls[call] = (frames, mp4)

    def _fill_s2():
        while s2_pending and len(s2_calls) < MODAL_MAX_INFLIGHT:
            wav_path, frames, mp4 = s2_pending.pop(0)
            _submit_s2(wav_path, frames, mp4)

    def _poll_s2():
        done = []
        for call, meta in list(s2_calls.items()):
            try:
                r2 = call.get(timeout=0)
                done.append((call, r2, meta))
            except TimeoutError:
                pass
            except Exception:
                done.append((call, {"text": "", "s2_time": 0}, meta))
        for call, r2, (frames, mp4) in done:
            s2_calls.pop(call, None)
            s2_times.append(r2["s2_time"])
            pbar2.update(1)
            s3f = clip.embed.remote(frames, r2["text"])
            s3_futures[s3f] = mp4
        _fill_s2()

    _submit_s1()

    pbar1 = tqdm(total=n_videos, desc=f"S1-FFmpeg ({S1_WORKERS:2d}w)", ncols=72,
                 colour="blue", position=0, leave=True)
    pbar2 = tqdm(total=n_videos,
                 desc=f"S2-Modal {MODAL_MAX_CONTAINERS}×{MODAL_GPU}",
                 ncols=72, colour="magenta", position=1, leave=True)
    pbar3 = tqdm(total=n_videos, desc=f"S3-CLIP   local ", ncols=72,
                 colour="green", position=2, leave=True)

    wall_start = time.time()

    while s1_futures or s1_queue or s2_pending or s2_calls or s3_futures:
        if s1_futures:
            done1, _ = ray.wait(list(s1_futures),
                                num_returns=min(8, len(s1_futures)), timeout=0.05)
            for f in done1:
                r1 = ray.get(f)
                s1_futures.pop(f)
                s1_times.append(r1["s1_time"])
                pbar1.update(1)
                if r1["wav"]:
                    s2_pending.append((r1["wav"], r1["frames"], r1["mp4"]))
                    _fill_s2()
            _submit_s1()

        if s2_calls:
            _poll_s2()

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
        "cpu_workers": num_workers,
        "s1_workers": S1_WORKERS,
        "n_videos": n_videos,
        "n_creators": n_creators,
        "s1_avg_s": round(np.mean(s1_times), 3) if s1_times else 0,
        "s2_avg_s": round(np.mean(s2_times), 3) if s2_times else 0,
        "s3_avg_s": round(np.mean(s3_times), 3) if s3_times else 0,
        "total_wall_s": total_wall,
        "throughput_vps": round(n_videos / total_wall, 3) if total_wall else 0,
        "avg_cpu_pct": round(np.mean([m["cpu_pct"] for m in _mon_log]), 1) if _mon_log else 0,
        "peak_gpu_mb": round(max((m["gpu_mb"] for m in _mon_log), default=0), 1),
        "modal_gpu": MODAL_GPU,
        "modal_containers": MODAL_MAX_CONTAINERS,
        "modal_parallel": MODAL_MAX_INFLIGHT,
    }


def plot_chart(results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workers = [r["cpu_workers"] for r in results]
    walls   = [r["total_wall_s"] for r in results]
    tputs   = [r["throughput_vps"] for r in results]
    cmap    = plt.cm.Blues(np.linspace(0.45, 0.95, len(workers)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Modal {MODAL_MAX_CONTAINERS}×{MODAL_GPU} Whisper + Local CLIP  |  {TARGET_VIDEOS} videos",
        fontsize=11, fontweight="bold", color="#e6edf3", y=1.02
    )
    plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
                         "text.color": "#e6edf3"})

    ax = axes[0]
    ax.bar([str(w) for w in workers], walls, color=cmap)
    ax.set_title("Wall time (s)"); ax.set_xlabel("S1 CPU workers")

    ax = axes[1]
    ax.bar([str(w) for w in workers], tputs, color=cmap)
    ax.set_title("Throughput (vid/s)"); ax.set_xlabel("S1 CPU workers")

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Chart saved: {OUT_CHART}")


def main():
    try:
        ray.shutdown()
    except Exception:
        pass

    n_cpus = min(LOGICAL_CPUS, os.cpu_count() or LOGICAL_CPUS)
    configs = [w for w in CPU_CONFIGS if w <= n_cpus] or [n_cpus]

    ray.init(address="local",
             num_cpus=n_cpus,
             num_gpus=1 if torch.cuda.is_available() else 0,
             ignore_reinit_error=True,
             runtime_env={"env_vars": {"OMP_NUM_THREADS": "1",
                                       "MKL_NUM_THREADS": "1"}})

    print(f"\n  Modal benchmark — S2: {MODAL_MAX_CONTAINERS}×{MODAL_GPU} "
          f"({MODAL_MAX_INFLIGHT} parallel), S1/S3 local")
    print(f"  S1 FFmpeg     : {S1_WORKERS} parallel workers ({FFMPEG_THREADS} thread each)")
    print(f"  S1 worker runs : {configs} (benchmark labels)")
    print(f"  Ray ready      |  {ray.cluster_resources()}\n")

    real_mp4s = []
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir) or creator.startswith("_"):
            continue
        real_mp4s.extend(sorted(glob.glob(os.path.join(cdir, "*.mp4"))))
    if not real_mp4s:
        print("  ERROR: No .mp4 files in results_4/")
        ray.shutdown(); return

    all_mp4s = (real_mp4s * ((TARGET_VIDEOS // len(real_mp4s)) + 1))[:TARGET_VIDEOS]
    audio_label = "full-length" if AUDIO_MAX_SEC is None else f"{AUDIO_MAX_SEC}s max"
    print(f"  Videos : {len(all_mp4s)}  |  Audio : {audio_label}")
    print(f"  Budget tip: watch usage at https://modal.com/settings\n")

    clip_actor = GPUCLIPActor.remote()
    try:
        ray.get(clip_actor.embed.remote([], ""), timeout=60)
    except Exception:
        pass

    bench_dir = os.path.join(RESULTS_DIR, "_bench_tmp_modal")
    results = []

    with app.run():
        modal_whisper = ModalWhisper()
        # Warmup Modal container
        try:
            modal_whisper.transcribe.remote(b"")
        except Exception:
            pass
        print("  Modal Whisper ready.\n")

        for n_workers in configs:
            print(f"\n{'='*60}")
            print(f"  RUN: {n_workers} S1 workers | {len(all_mp4s)} videos | "
                  f"Modal {MODAL_MAX_CONTAINERS}×{MODAL_GPU}")
            print(f"{'='*60}")

            start_monitor()
            try:
                row = run_pipeline(n_workers, all_mp4s, modal_whisper, clip_actor, bench_dir)
            except Exception as e:
                stop_monitor()
                print(f"  [SKIP] failed: {e}")
                shutil.rmtree(bench_dir, ignore_errors=True)
                continue
            stop_monitor()
            results.append(row)
            print(f"\n  Wall {row['total_wall_s']}s | {row['throughput_vps']} vid/s | "
                  f"S1 {row['s1_avg_s']}s  S2 {row['s2_avg_s']}s  S3 {row['s3_avg_s']}s")

    if not results:
        ray.shutdown(); return

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")
    print(df.to_string(index=False))
    plot_chart(results)
    ray.shutdown()


if __name__ == "__main__":
    main()
