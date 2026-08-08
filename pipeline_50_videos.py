"""
pipeline_50_videos.py — Streaming zero-disk-I/O Ray pipeline
=============================================================
Producer-consumer architecture with bounded memory:

  S1 (CPU) ──batch──▶ S2 (GPU Whisper) ──batch──▶ S3 (GPU CLIP) ──▶ FAISS

  • Audio stays as s16le PCM bytes until inside the Whisper worker
  • Frames stay as JPEG bytes; no .wav / image files written
  • Ray object store never holds all videos at once
  • S1 workers scale horizontally across CPU nodes

Run:
  python pipeline_50_videos.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
import ray
import torch
from tqdm import tqdm

from config import (
    RESULTS_DIR, TRANSCRIPTIONS_DIR, WHISPER_BATCH_SIZE,
    S2_MAX_INFLIGHT, S3_CREATOR_BATCH, S3_MAX_INFLIGHT,
    EMBEDDER_GPU_FRAC, GPU_RELEASE_TIMEOUT, RAY_ADDRESS,
)
from utils import ensure_dir
from stage1_download_extract import extract_video_to_memory_remote
from stage2_transcribe import GPUWhisperActor, aggregate_creator_transcriptions
from stage3_embeddings import GPUBatchEmbeddingActor

TARGET_VIDEOS = 948  # all videos already downloaded in RESULTS_DIR (no repeats)
S1_CONFIGS = [8]  # single run to regenerate creators.txt/image_embs.npy/text_embs.npy at full scale
OUT_BENCHMARK_CSV = "pipeline_streaming_benchmark_results_vid_9.csv"
OUT_TRANSCRIPTIONS = os.path.join(TRANSCRIPTIONS_DIR, "pipeline_streaming_transcriptions.csv")


def _dbg(msg: str):
    tqdm.write(f"[pipeline] {msg}")


_mon_active = False
_mon_log: list = []


def _start_cpu_monitor():
    global _mon_active, _mon_log
    _mon_active, _mon_log = True, []

    def _loop():
        while _mon_active:
            _mon_log.append({"cpu_pct": psutil.cpu_percent(interval=1)})
            time.sleep(1)

    threading.Thread(target=_loop, daemon=True).start()


def _stop_cpu_monitor() -> float:
    global _mon_active
    _mon_active = False
    time.sleep(0.3)
    if not _mon_log:
        return 0.0
    return round(float(np.mean([m["cpu_pct"] for m in _mon_log])), 1)


def _print_scalability_table(results: list[dict]):
    headers = ["S1 Workers", "Stage 2 avg (s)", "Stage 3 avg (s)",
               "Total wall (s)", "Speedup", "Avg CPU %"]
    tqdm.write("\nSCALABILITY BENCHMARK:")
    tqdm.write("  ".join(f"{h:>16}" for h in headers))
    for row in results:
        tqdm.write(
            f"{row['s1_workers']:16d}"
            f"{row['stage2_avg_s']:16.2f}"
            f"{row['stage3_avg_s']:16.2f}"
            f"{row['total_wall_s']:16.2f}"
            f"{row['speedup']:15.2f}x"
            f"{row['avg_cpu_pct']:15.1f}%"
        )


def _gpu_available() -> float:
    return ray.available_resources().get("GPU", 0.0)


def _wait_for_gpu(needed: float, timeout: float = GPU_RELEASE_TIMEOUT) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        avail = _gpu_available()
        if avail >= needed:
            _dbg(f"GPU available: {avail:.2f} (need {needed:.2f})")
            return True
        time.sleep(0.5)
    _dbg(f"GPU wait timeout — available={_gpu_available():.2f}, need={needed:.2f}")
    return False


def _transition_s2_to_s3(whisper) -> object:
    """Gracefully release Whisper GPU memory, then spawn CLIP actor."""
    _dbg("S2 complete — shutting down Whisper actor...")
    try:
        shutdown_info = ray.get(whisper.shutdown.remote(), timeout=60)
        _dbg(
            f"Whisper shutdown: {shutdown_info.get('shutdown_s')}s, "
            f"VRAM {shutdown_info.get('vram_before_mb')}→"
            f"{shutdown_info.get('vram_after_mb')} MB"
        )
    except Exception as e:
        _dbg(f"Whisper shutdown remote failed: {e}")

    ray.kill(whisper, no_restart=True)
    _dbg("Whisper actor killed — waiting for GPU release...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _dbg(f"Driver VRAM after cleanup: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    if not _wait_for_gpu(EMBEDDER_GPU_FRAC):
        _dbg("WARNING: proceeding anyway — CLIP may block on GPU scheduling")

    t0 = time.time()
    embedder = GPUBatchEmbeddingActor.remote()
    warmup = ray.get(embedder.warmup.remote(), timeout=120)
    _dbg(
        f"CLIP actor ready in {time.time()-t0:.1f}s "
        f"(model init {warmup.get('init_s')}s, VRAM {warmup.get('vram_mb')} MB)"
    )
    return embedder


def collect_videos(target: int = TARGET_VIDEOS) -> list[str]:
    real_mp4s = []
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir) or creator.startswith("_"):
            continue
        real_mp4s.extend(sorted(glob.glob(os.path.join(cdir, "*.mp4"))))
    if not real_mp4s:
        return []
    n = len(real_mp4s)
    return (real_mp4s * ((target // n) + 1))[:target]


def run_streaming_pipeline(videos: list[str], s1_workers: int) -> dict:
    """
    Bounded-memory streaming pipeline: S1 produces → S2 consumes → S3 consumes.
    Audio PCM is released after each Whisper batch; only texts + JPEG bytes retained.
    """
    n_videos = len(videos)
    creators = sorted({os.path.basename(os.path.dirname(p)) for p in videos})

    creator_videos: dict[str, list[str]] = {}
    for mp4 in videos:
        c = os.path.basename(os.path.dirname(mp4))
        creator_videos.setdefault(c, []).append(mp4)

    creator_needed = {c: len(vids) for c, vids in creator_videos.items()}
    creator_done_count: dict[str, int] = defaultdict(int)
    creator_mp4_texts: dict[str, dict[str, str]] = defaultdict(dict)
    creator_frame_bytes: dict[str, list[bytes]] = defaultdict(list)

    video_queue = list(videos)
    s1_futures: dict = {}
    s2_pending: list[dict] = []
    s2_futures: dict = {}
    s3_pending: list[dict] = []
    s3_futures: dict = {}
    s3_results: list = []

    video_texts: dict[str, str] = {}
    s1_time_sum = 0.0
    s2_times: list[float] = []
    s3_times: list[float] = []
    s3_submit_meta: dict = {}
    s2_active_start = None
    s2_active_end = 0.0
    s3_active_start = None
    s3_active_end = 0.0
    videos_s2_done = 0
    creators_s3_done = 0

    whisper = GPUWhisperActor.remote()
    ray.get(whisper.warmup.remote())
    embedder = None

    def _submit_s1():
        while video_queue and len(s1_futures) < s1_workers:
            mp4 = video_queue.pop(0)
            fut = extract_video_to_memory_remote.remote(mp4)
            s1_futures[fut] = mp4

    def _fill_s2():
        nonlocal s2_active_start
        while len(s2_pending) >= WHISPER_BATCH_SIZE and len(s2_futures) < S2_MAX_INFLIGHT:
            batch = s2_pending[:WHISPER_BATCH_SIZE]
            del s2_pending[:WHISPER_BATCH_SIZE]
            if s2_active_start is None:
                s2_active_start = time.time()
            fut = whisper.transcribe_batch_memory.remote(batch)
            s2_futures[fut] = len(batch)

    def _flush_s2():
        nonlocal s2_active_start
        if s2_pending and len(s2_futures) < S2_MAX_INFLIGHT:
            batch = s2_pending[:]
            s2_pending.clear()
            if s2_active_start is None:
                s2_active_start = time.time()
            _dbg(f"S2 flush: {len(batch)} videos (partial/final batch)")
            fut = whisper.transcribe_batch_memory.remote(batch)
            s2_futures[fut] = len(batch)

    def _maybe_flush_s2_partial():
        """Flush remainder when S1 is done and pending < WHISPER_BATCH_SIZE."""
        s1_done = not video_queue and not s1_futures
        if (s1_done and s2_pending
                and len(s2_pending) < WHISPER_BATCH_SIZE
                and len(s2_futures) < S2_MAX_INFLIGHT):
            _flush_s2()

    def _on_s2_result(result: dict, batch_size: int):
        nonlocal s2_active_end, videos_s2_done
        s2_active_end = time.time()
        videos_s2_done += batch_size
        n_files = max(result.get("n_files", batch_size), 1)
        s2_times.append(result.get("s2_time", 0) / n_files)

        for mp4, text in result["texts"].items():
            video_texts[mp4] = text
            creator = os.path.basename(os.path.dirname(mp4))
            creator_mp4_texts[creator][mp4] = text
            creator_frame_bytes[creator].extend(result.get("frames", {}).get(mp4, []))
            creator_done_count[creator] += 1

            if creator_done_count[creator] == creator_needed[creator]:
                full_text = " ".join(
                    creator_mp4_texts[creator].get(p, "")
                    for p in sorted(creator_videos[creator])
                )
                s3_pending.append({
                    "creator": creator,
                    "text": full_text,
                    "frames": creator_frame_bytes.pop(creator, []),
                })
                del creator_mp4_texts[creator]
                _dbg(
                    f"creator ready for S3: {creator} "
                    f"({len(s3_pending)} queued / {len(creators)} total)"
                )

    def _submit_s3():
        nonlocal embedder, s3_active_start
        if embedder is None or not s3_pending:
            return
        while s3_pending and len(s3_futures) < S3_MAX_INFLIGHT:
            batch = s3_pending[:S3_CREATOR_BATCH]
            del s3_pending[:len(batch)]
            if s3_active_start is None:
                s3_active_start = time.time()
            _dbg(f"S3 submit: {len(batch)} creators, queue_left={len(s3_pending)}")
            fut = embedder.embed_creators_batch.remote(batch)
            s3_futures[fut] = len(batch)
            s3_submit_meta[fut] = (time.time(), len(batch))

    def _flush_s3():
        nonlocal embedder, s3_active_start
        if embedder is None or not s3_pending:
            return
        if len(s3_futures) >= S3_MAX_INFLIGHT:
            return
        batch = s3_pending[:]
        s3_pending.clear()
        if s3_active_start is None:
            s3_active_start = time.time()
        _dbg(f"S3 flush: {len(batch)} creators (final batch)")
        fut = embedder.embed_creators_batch.remote(batch)
        s3_futures[fut] = len(batch)
        s3_submit_meta[fut] = (time.time(), len(batch))

    _submit_s1()

    pbar1 = tqdm(total=n_videos, desc=f"S1→mem ({s1_workers}w)", ncols=78, colour="blue", position=0)
    pbar2 = tqdm(total=n_videos, desc=f"S2 Whisper (batch={WHISPER_BATCH_SIZE})", ncols=78,
                 colour="magenta", position=1)
    pbar3 = tqdm(total=len(creators), desc="S3 CLIP (streaming)", ncols=78, colour="green", position=2)

    wall_start = time.time()

    while (video_queue or s1_futures or s2_pending or s2_futures
           or s3_pending or s3_futures or videos_s2_done < n_videos
           or creators_s3_done < len(creators)):

        if s1_futures:
            done, _ = ray.wait(list(s1_futures), num_returns=min(8, len(s1_futures)), timeout=0.05)
            for f in done:
                payload = ray.get(f)
                s1_futures.pop(f)
                s1_time_sum += payload.get("s1_time", 0)
                pbar1.update(1)
                if payload.get("has_audio"):
                    s2_pending.append(payload)
                else:
                    mp4 = payload["mp4"]
                    creator = payload["creator"]
                    video_texts[mp4] = ""
                    creator_mp4_texts[creator][mp4] = ""
                    creator_done_count[creator] += 1
                    videos_s2_done += 1
                    pbar2.update(1)
                    if creator_done_count[creator] == creator_needed[creator]:
                        full_text = " ".join(
                            creator_mp4_texts[creator].get(p, "")
                            for p in sorted(creator_videos[creator])
                        )
                        s3_pending.append({
                            "creator": creator,
                            "text": full_text,
                            "frames": creator_frame_bytes.pop(creator, []),
                        })
                        del creator_mp4_texts[creator]
                        _dbg(
                            f"creator ready for S3 (no-audio): {creator} "
                            f"({len(s3_pending)} queued)"
                        )
                _fill_s2()
                _maybe_flush_s2_partial()
            _submit_s1()

        _fill_s2()
        _maybe_flush_s2_partial()

        if s2_futures:
            done, _ = ray.wait(list(s2_futures), num_returns=min(2, len(s2_futures)), timeout=0.05)
            for f in done:
                batch_size = s2_futures.pop(f)
                result = ray.get(f)
                _on_s2_result(result, batch_size)
                pbar2.update(batch_size)
                _fill_s2()
                _maybe_flush_s2_partial()

        if (videos_s2_done >= n_videos and not s2_futures and not s2_pending
                and embedder is None):
            _dbg(
                f"S2 done ({videos_s2_done}/{n_videos} videos), "
                f"S3 queue={len(s3_pending)} creators — transitioning GPU..."
            )
            embedder = _transition_s2_to_s3(whisper)
            _submit_s3()

        if embedder is not None:
            _submit_s3()
            if s3_futures:
                done, _ = ray.wait(list(s3_futures), num_returns=1, timeout=0.1)
                for f in done:
                    n = s3_futures.pop(f)
                    t0, batch_n = s3_submit_meta.pop(f, (time.time(), n))
                    s3_results.extend(ray.get(f))
                    s3_times.append((time.time() - t0) / max(batch_n, 1))
                    s3_active_end = time.time()
                    creators_s3_done += n
                    pbar3.update(n)
                    _dbg(
                        f"S3 batch done: +{n} creators "
                        f"({creators_s3_done}/{len(creators)}), "
                        f"queue={len(s3_pending)}, inflight={len(s3_futures)}"
                    )
            if (videos_s2_done >= n_videos and not s2_futures and not s2_pending
                    and s3_pending and len(s3_futures) < S3_MAX_INFLIGHT):
                _flush_s3()

    pbar1.close()
    pbar2.close()
    pbar3.close()

    transcriptions = aggregate_creator_transcriptions(
        video_texts, creator_videos, creators
    )

    if embedder and s3_results:
        ray.get(embedder.build_faiss_index.remote(s3_results, transcriptions))
        ray.kill(embedder)

    total_wall = time.time() - wall_start
    s2_wall = (s2_active_end - s2_active_start) if s2_active_start else 0
    s3_wall = (s3_active_end - s3_active_start) if s3_active_start else 0
    stage2_avg_s = round(float(np.mean(s2_times)), 2) if s2_times else 0
    stage3_avg_s = round(float(np.mean(s3_times)), 2) if s3_times else 0

    return {
        "s1_workers": s1_workers,
        "n_videos": n_videos,
        "n_creators": len(creators),
        "stage2_avg_s": stage2_avg_s,
        "stage3_avg_s": stage3_avg_s,
        "total_wall_s": round(total_wall, 2),
        "stage_1_time_sec": round(s1_time_sum, 2),
        "stage_2_time_sec": round(s2_wall, 2),
        "stage_3_time_sec": round(s3_wall, 2),
        "total_time_sec": round(total_wall, 2),
        "stage_1_vps": round(n_videos / s1_time_sum, 3) if s1_time_sum else 0,
        "stage_2_vps": round(n_videos / s2_wall, 3) if s2_wall else 0,
        "stage_3_vps": round(n_videos / s3_wall, 3) if s3_wall else 0,
        "total_vps": round(n_videos / total_wall, 3) if total_wall else 0,
        "transcriptions": transcriptions,
    }


def main():
    try:
        ray.shutdown()
    except Exception:
        pass

    max_cpus = max(S1_CONFIGS)
    init_kwargs = dict(
        num_cpus=max_cpus + 2,
        num_gpus=1 if torch.cuda.is_available() else 0,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    )
    if RAY_ADDRESS:
        ray.init(address=RAY_ADDRESS, **init_kwargs)
    else:
        ray.init(address="local", **init_kwargs)

    videos = collect_videos()
    if not videos:
        tqdm.write(f"No .mp4 files in {RESULTS_DIR}/")
        ray.shutdown()
        return

    tqdm.write("\nStreaming pipeline — bounded Ray memory, zero intermediate disk I/O\n")
    tqdm.write(f"Ray cluster: {ray.cluster_resources()}\n")

    results = []
    baseline_wall = None
    for i, s1_workers in enumerate(S1_CONFIGS):
        tqdm.write(f"--- Run {i + 1}/{len(S1_CONFIGS)}: S1 workers = {s1_workers} ---")
        _start_cpu_monitor()
        row = run_streaming_pipeline(videos, s1_workers)
        row["avg_cpu_pct"] = _stop_cpu_monitor()
        if baseline_wall is None:
            baseline_wall = row["total_wall_s"]
        row["speedup"] = round(baseline_wall / row["total_wall_s"], 2) if row["total_wall_s"] else 0
        results.append({k: v for k, v in row.items() if k != "transcriptions"})

        tqdm.write(
            f"  Stage2 avg {row['stage2_avg_s']}s/video | "
            f"Stage3 avg {row['stage3_avg_s']}s/creator | "
            f"Total {row['total_wall_s']}s | "
            f"x{row['speedup']} | "
            f"CPU avg {row['avg_cpu_pct']}%"
        )

        if i == len(S1_CONFIGS) - 1:
            ensure_dir(TRANSCRIPTIONS_DIR)
            pd.DataFrame(
                list(row["transcriptions"].items()),
                columns=["creator", "transcription"],
            ).to_csv(OUT_TRANSCRIPTIONS, index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUT_BENCHMARK_CSV, index=False)

    _print_scalability_table(results)
    tqdm.write("\nFULL BENCHMARK CSV:")
    tqdm.write(df.to_string(index=False))
    tqdm.write(f"\nSaved: {OUT_BENCHMARK_CSV}")
    ray.shutdown()


if __name__ == "__main__":
    main()
