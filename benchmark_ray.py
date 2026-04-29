"""
benchmark_ray.py  —  Concurrent Distributed Ray Pipeline Benchmark
===================================================================
Pipeline (all stages overlap):
  Stage 2  →  Whisper per video   (CPU workers — scales)
  Stage 3  →  CLIP per creator    (GPU actor   — fires as soon as creator's
                                    3 videos finish Stage 2)

Workers tested : 1 / 2 / 4 / 8  CPU workers
Creators       : 3  (3 videos each = 9 Stage-2 tasks)

Run:   python benchmark_ray.py
Dashboard:     http://localhost:8265
Output:        benchmark_results.csv
"""

import os, glob, time, threading
import numpy as np, pandas as pd
import ray, psutil, torch


# ── Stage 2: CPU Whisper — one worker transcribes one wav file ────────────────
@ray.remote(num_cpus=1)
class CPUWhisperWorker:
    def __init__(self, worker_id: int):
        from faster_whisper import WhisperModel
        self.wid = worker_id
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print(f"  [Whisper-{worker_id}] ready on CPU")

    def transcribe(self, wav_path: str) -> dict:
        """Transcribe one wav file. Returns timing + text."""
        t0 = time.time()
        try:
            segs, _ = self.model.transcribe(
                wav_path, beam_size=1, vad_filter=False, without_timestamps=True
            )
            text = " ".join(s.text.strip() for s in segs)
        except Exception as e:
            text = ""
            print(f"  [Whisper-{self.wid}] ⚠️  {wav_path}: {e}")
        elapsed = round(time.time() - t0, 3)
        print(f"  [Whisper-{self.wid}] {os.path.basename(wav_path)} → {elapsed:.1f}s")
        return {"wav": wav_path, "text": text, "stage2_time": elapsed, "worker": self.wid}


# ── Stage 3: GPU CLIP — embed one creator ─────────────────────────────────────
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
        print(f"  [CLIP-GPU] ready on {self.device.upper()}")

    def embed(self, creator: str, transcription: str, results_dir: str) -> dict:
        from PIL import Image
        t0 = time.time()
        frames = glob.glob(
            os.path.join(results_dir, creator, "**/frames/*.*"), recursive=True
        )
        frames = [f for f in frames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        tensors = []
        for f in frames[:10]:
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
        elapsed = round(time.time() - t0, 3)
        print(f"  [CLIP-GPU] {creator} → {elapsed:.1f}s")
        return {"creator": creator, "stage3_time": elapsed}


# ── Live monitor ──────────────────────────────────────────────────────────────
_mon_active = False
_mon_log: list = []

def _mon_thread():
    while _mon_active:
        snap = {
            "cpu_pct": psutil.cpu_percent(interval=1),
            "ram_gb":  round(psutil.virtual_memory().used / 1e9, 2),
            "gpu_mb":  round(torch.cuda.memory_allocated() / 1e6, 1)
                       if torch.cuda.is_available() else 0,
        }
        _mon_log.append(snap)
        print(f"  📡  CPU {snap['cpu_pct']:5.1f}%  "
              f"RAM {snap['ram_gb']:.1f} GB  "
              f"GPU {snap['gpu_mb']:.0f} MB")
        time.sleep(2)

def start_monitor():
    global _mon_active, _mon_log
    _mon_active = True
    _mon_log = []
    threading.Thread(target=_mon_thread, daemon=True).start()

def stop_monitor():
    global _mon_active
    _mon_active = False


# ── One full concurrent pipeline run ─────────────────────────────────────────
def run_pipeline(num_cpu_workers: int, creators: list, results_dir: str) -> dict:
    """
    Concurrent pipeline:
    - All Stage-2 tasks submitted at once across N workers.
    - ray.wait() loop: as soon as a creator's last video finishes Stage 2,
      Stage 3 fires immediately — no waiting for other creators.
    """
    # Build wav list per creator (max 3 videos each)
    creator_wavs: dict[str, list] = {}
    for c in creators:
        wavs = glob.glob(os.path.join(results_dir, c, "**/*.wav"), recursive=True)
        creator_wavs[c] = wavs[:3]

    # Spin up N CPU workers (round-robin assignment)
    workers = [CPUWhisperWorker.remote(i) for i in range(num_cpu_workers)]
    gpu_actor = GPUEmbeddingActor.remote()

    # Submit all Stage-2 tasks
    future_meta: dict = {}   # future_id -> (creator, wav_path)
    worker_idx = 0
    for c, wavs in creator_wavs.items():
        for w in wavs:
            fut = workers[worker_idx % num_cpu_workers].transcribe.remote(w)
            future_meta[fut] = (c, w)
            worker_idx += 1

    # Track per-creator completion
    creator_texts: dict[str, list]  = {c: [] for c in creators}
    creator_s2_times: dict[str, list] = {c: [] for c in creators}
    creator_expected = {c: len(v) for c, v in creator_wavs.items()}
    stage3_futures: dict = {}   # future -> creator

    wall_start = time.time()
    remaining_s2 = list(future_meta.keys())

    # ── Concurrent streaming loop ─────────────────────────────────────────────
    while remaining_s2:
        done, remaining_s2 = ray.wait(remaining_s2, num_returns=1, timeout=None)
        for fut in done:
            result = ray.get(fut)
            creator, _ = future_meta[fut]
            creator_texts[creator].append(result["text"])
            creator_s2_times[creator].append(result["stage2_time"])

            # Fire Stage 3 as soon as all videos for this creator are done
            if len(creator_texts[creator]) == creator_expected[creator]:
                combined = " ".join(creator_texts[creator])
                s3_fut = gpu_actor.embed.remote(creator, combined, results_dir)
                stage3_futures[s3_fut] = creator

    # ── Wait for all Stage-3 tasks ────────────────────────────────────────────
    s3_results = ray.get(list(stage3_futures.keys()))
    total_wall = round(time.time() - wall_start, 2)

    # ── Aggregate timings ─────────────────────────────────────────────────────
    avg_s2 = round(np.mean([t for ts in creator_s2_times.values() for t in ts]), 2)
    avg_s3 = round(np.mean([r["stage3_time"] for r in s3_results]), 2)

    return {
        "cpu_workers":    num_cpu_workers,
        "creators":       len(creators),
        "videos":         sum(creator_expected.values()),
        "stage2_avg_s":   avg_s2,
        "stage3_avg_s":   avg_s3,
        "total_wall_s":   total_wall,
        "throughput_c_s": round(len(creators) / total_wall, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR = "results_4"

    ray.init(
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
        num_gpus=1 if torch.cuda.is_available() else 0,
    )
    print(f"\n🚀 Ray ready | {ray.cluster_resources()}")
    print(f"📊 Ray Dashboard → http://localhost:8265\n")

    # Pick 3 creators with ≥3 wav files
    creators = [
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and len(glob.glob(os.path.join(RESULTS_DIR, d, "**/*.wav"), recursive=True)) >= 3
    ][:3]
    print(f"📦 Creators selected: {creators}\n")

    results = []
    baseline_wall = None

    for n in [1, 2, 4, 8]:
        print(f"\n{'='*60}")
        print(f"  RUN: {n} CPU worker(s)  (concurrent Stage2 + Stage3)")
        print(f"{'='*60}")
        start_monitor()
        row = run_pipeline(n, creators, RESULTS_DIR)
        stop_monitor()
        time.sleep(0.5)   # let monitor flush

        if baseline_wall is None:
            baseline_wall = row["total_wall_s"]
        row["speedup"] = round(baseline_wall / row["total_wall_s"], 2)

        avg_cpu = round(np.mean([m["cpu_pct"] for m in _mon_log]), 1) if _mon_log else 0
        peak_gpu = round(max((m["gpu_mb"] for m in _mon_log), default=0), 1)
        row["avg_cpu_pct"] = avg_cpu
        row["peak_gpu_mb"] = peak_gpu

        results.append(row)
        print(f"\n  ✅  {n} worker(s) | "
              f"Stage2 avg {row['stage2_avg_s']}s | "
              f"Stage3 avg {row['stage3_avg_s']}s | "
              f"Total {row['total_wall_s']}s | "
              f"Speedup ×{row['speedup']}")

    # ── Final table ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    cols = ["cpu_workers", "stage2_avg_s", "stage3_avg_s",
            "total_wall_s", "speedup", "avg_cpu_pct", "peak_gpu_mb"]

    print(f"\n\n{'='*70}")
    print("  FINAL BENCHMARK — Concurrent Ray Pipeline (3 creators × 3 videos)")
    print(f"{'='*70}")
    print(df[cols].to_string(index=False))
    print(f"{'='*70}")
    print("\nNotes:")
    print("  • Stage 2 (Whisper) runs on CPU → scales with more workers")
    print("  • Stage 3 (CLIP)    runs on GPU → fires per-creator as soon as Stage 2 done")
    print("  • Stages overlap: GPU is never idle waiting for all CPU work to finish")
    print(f"{'='*70}")

    df[cols].to_csv("benchmark_results.csv", index=False)
    print("\n💾 Saved → benchmark_results.csv")

    ray.shutdown()


if __name__ == "__main__":
    main()
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MULTI-SERVER SETUP — what changes when you have real nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

On HEAD node (your main machine):
  ray start --head --port=6379 --dashboard-host=0.0.0.0

On each WORKER node (other servers):
  ray start --address='<HEAD_IP>:6379'
  (needs Python + same packages installed: pip install -r requirements.txt)

In this file — change ONE line:
  ray.init(address='auto')   ← instead of ray.init(include_dashboard=True, ...)

That is it. Ray automatically distributes @ray.remote tasks across all nodes.
num_cpu_workers becomes the TOTAL CPUs across all machines.

Shared data (results_4/, wav files) must be on:
  - A shared network drive (NFS/SMB) mounted at the same path on every node, OR
  - Object storage (S3/MinIO) with paths updated in config.py

So currently: 8 "workers" = 8 CPU cores on 1 laptop  (simulated nodes)
With 2 real servers (8 cores each): 8 workers = 4 per server  (real distribution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
