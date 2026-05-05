"""
scalability_benchmark.py  —  Full Scalability Matrix
=====================================================
Tests: 10 / 20 / 30 videos  ×  1 / 2 / 4 / 8 CPU workers  = 12 runs

Live progress:
  Terminal  → tqdm progress bar per run
  Browser   → http://localhost:8888  (auto-refreshes every 3s)

Output: scalability_results.csv
"""

import os, glob, time, json, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np, pandas as pd
from tqdm import tqdm
import ray, psutil, torch


# ── Shared progress state (written by benchmark, read by web server) ──────────
_state = {
    "current_run":   "",
    "run_index":     0,
    "total_runs":    12,
    "videos_done":   0,
    "videos_total":  0,
    "stage":         "idle",
    "results":       [],
    "done":          False,
}
_state_lock = threading.Lock()

def update_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)

def get_state():
    with _state_lock:
        return dict(_state)


# ── Tiny web dashboard ────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Ray Scalability Benchmark</title>
  <meta http-equiv="refresh" content="3">
  <style>
    body {{ font-family: monospace; background:#0d1117; color:#e6edf3; padding:30px; }}
    h1   {{ color:#58a6ff; }}
    .box {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
            padding:20px; margin:16px 0; }}
    .label {{ color:#8b949e; font-size:13px; }}
    .val   {{ font-size:22px; font-weight:bold; color:#3fb950; }}
    .bar-bg  {{ background:#21262d; border-radius:4px; height:22px; width:100%; }}
    .bar-fg  {{ background:#238636; height:22px; border-radius:4px;
                transition:width 0.4s; }}
    table  {{ border-collapse:collapse; width:100%; }}
    th     {{ background:#21262d; padding:8px 12px; color:#58a6ff; text-align:left; }}
    td     {{ padding:6px 12px; border-bottom:1px solid #21262d; }}
    tr:hover td {{ background:#161b22; }}
    .done  {{ color:#f78166; font-size:18px; font-weight:bold; }}
  </style>
</head>
<body>
  <h1>🚀 Ray Scalability Benchmark — Live</h1>
  <div class="box">
    <div class="label">Current run</div>
    <div class="val">{current_run}</div>
    <br>
    <div class="label">Overall progress ({run_index} / {total_runs} runs)</div>
    <div class="bar-bg"><div class="bar-fg" style="width:{run_pct}%"></div></div>
    <br>
    <div class="label">Videos in this run ({videos_done} / {videos_total})</div>
    <div class="bar-bg"><div class="bar-fg" style="width:{vid_pct}%"></div></div>
    <br>
    <div class="label">Stage</div>
    <div class="val" style="font-size:16px">{stage}</div>
  </div>

  {results_table}

  <div class="label" style="margin-top:20px">
    Auto-refreshes every 3s &nbsp;|&nbsp;
    Ray Dashboard → <a href="http://localhost:8265" style="color:#58a6ff">localhost:8265</a>
  </div>
  {done_banner}
</body>
</html>"""

def build_html(s):
    run_pct = round(s["run_index"] / s["total_runs"] * 100) if s["total_runs"] else 0
    vid_pct = round(s["videos_done"] / s["videos_total"] * 100) if s["videos_total"] else 0

    rows = ""
    if s["results"]:
        rows = "<tr>" + "".join(
            f"<th>{c}</th>" for c in
            ["Workers","Videos","Stage2 avg(s)","Stage3 avg(s)","Total(s)","Speedup","CPU%"]
        ) + "</tr>"
        for r in s["results"]:
            rows += (
                f"<tr><td>{r['cpu_workers']}</td><td>{r['videos']}</td>"
                f"<td>{r['stage2_avg_s']}</td><td>{r['stage3_avg_s']}</td>"
                f"<td>{r['total_wall_s']}</td><td>×{r['speedup']}</td>"
                f"<td>{r['avg_cpu_pct']}%</td></tr>"
            )
    table = f'<div class="box"><table>{rows}</table></div>' if rows else ""
    done  = '<div class="done">✅ Benchmark complete — see scalability_results.csv</div>' \
            if s["done"] else ""
    return HTML_TEMPLATE.format(
        current_run=s["current_run"] or "—",
        run_index=s["run_index"],
        total_runs=s["total_runs"],
        run_pct=run_pct,
        videos_done=s["videos_done"],
        videos_total=s["videos_total"],
        vid_pct=vid_pct,
        stage=s["stage"],
        results_table=table,
        done_banner=done,
    )

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html(get_state()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)
    def log_message(self, *_):   # silence access log
        pass

def start_web_server(port=8888):
    srv = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"🌐 Live dashboard → http://localhost:{port}\n")


# ── Ray workers ───────────────────────────────────────────────────────────────
@ray.remote(num_cpus=1)
class CPUWhisperWorker:
    def __init__(self, wid: int):
        from faster_whisper import WhisperModel
        self.wid = wid
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str) -> dict:
        t0 = time.time()
        try:
            segs, _ = self.model.transcribe(
                wav_path, beam_size=1, vad_filter=False, without_timestamps=True
            )
            text = " ".join(s.text.strip() for s in segs)
        except Exception:
            text = ""
        return {"wav": wav_path, "text": text, "stage2_time": round(time.time()-t0, 3)}

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

    def embed(self, creator: str, transcription: str, results_dir: str) -> dict:
        from PIL import Image
        t0 = time.time()
        frames = glob.glob(
            os.path.join(results_dir, creator, "**/frames/*.*"), recursive=True
        )
        frames = [f for f in frames if f.lower().endswith((".png",".jpg",".jpeg"))]
        tensors = []
        for f in frames[:10]:
            try:
                tensors.append(self.preprocess(Image.open(f).convert("RGB")))
            except Exception:
                pass
        if tensors:
            with torch.no_grad():
                b = torch.stack(tensors).to(self.device)
                if self.device == "cuda": b = b.half()
                self.model.encode_image(b)
        if transcription.strip():
            with torch.no_grad():
                self.model.encode_text(
                    self.tokenizer([transcription[:77]]).to(self.device)
                )
        return {"creator": creator, "stage3_time": round(time.time()-t0, 3)}


# ── Single pipeline run ───────────────────────────────────────────────────────
def run_pipeline(num_workers: int, wav_list: list,
                 creator_of: dict, results_dir: str,
                 gpu_actor, pbar: tqdm) -> dict:
    """
    wav_list   : list of wav paths for this run
    creator_of : {wav_path: creator_name}
    """
    workers = [CPUWhisperWorker.remote(i) for i in range(num_workers)]

    # Submit all Stage-2 tasks (round-robin)
    future_meta = {}
    for i, w in enumerate(wav_list):
        fut = workers[i % num_workers].transcribe.remote(w)
        future_meta[fut] = w

    # Track per-creator completion
    unique_creators = list(set(creator_of[w] for w in wav_list))
    expected = {c: sum(1 for w in wav_list if creator_of[w] == c)
                for c in unique_creators}
    texts   = {c: [] for c in unique_creators}
    s2times = {c: [] for c in unique_creators}
    stage3_futures = {}

    remaining = list(future_meta.keys())
    update_state(videos_done=0, videos_total=len(wav_list), stage="Stage 2 — Whisper CPU")

    wall_start = time.time()
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        res = ray.get(done[0])
        creator = creator_of[res["wav"]]
        texts[creator].append(res["text"])
        s2times[creator].append(res["stage2_time"])
        pbar.update(1)
        update_state(videos_done=pbar.n)

        if len(texts[creator]) == expected[creator]:
            update_state(stage=f"Stage 3 — CLIP GPU ({creator})")
            combined = " ".join(texts[creator])
            s3f = gpu_actor.embed.remote(creator, combined, results_dir)
            stage3_futures[s3f] = creator

    update_state(stage="Stage 3 — CLIP GPU (finalizing)")
    s3_results = ray.get(list(stage3_futures.keys()))
    total_wall = round(time.time() - wall_start, 2)

    avg_s2 = round(np.mean([t for ts in s2times.values() for t in ts]), 2)
    avg_s3 = round(np.mean([r["stage3_time"] for r in s3_results]), 2)
    return {"stage2_avg_s": avg_s2, "stage3_avg_s": avg_s3, "total_wall_s": total_wall}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR  = "results_4"
    VIDEO_COUNTS = [10, 20, 30]
    WORKER_COUNTS = [1, 2, 4, 8]

    # ── Init ──────────────────────────────────────────────────────────────────
    start_web_server()
    ray.init(
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
        num_gpus=1 if torch.cuda.is_available() else 0,
    )
    print(f"🚀 Ray ready | {ray.cluster_resources()}")
    print(f"📊 Ray Dashboard → http://localhost:8265\n")

    # ── Build wav pool ─────────────────────────────────────────────────────────
    all_wavs, creator_of = [], {}
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir): continue
        wavs = glob.glob(os.path.join(cdir, "**/*.wav"), recursive=True)[:3]
        for w in wavs:
            all_wavs.append(w)
            creator_of[w] = creator

    if len(all_wavs) < max(VIDEO_COUNTS):
        print(f"⚠️  Only {len(all_wavs)} wav files found; "
              f"capping max video count to {len(all_wavs)}")
        VIDEO_COUNTS = [v for v in VIDEO_COUNTS if v <= len(all_wavs)]

    # ── GPU actor (shared across all runs) ────────────────────────────────────
    gpu_actor = GPUEmbeddingActor.remote()

    total_runs = len(VIDEO_COUNTS) * len(WORKER_COUNTS)
    update_state(total_runs=total_runs)

    all_results = []
    run_idx = 0

    for n_videos in VIDEO_COUNTS:
        wav_subset = all_wavs[:n_videos]
        baseline_time = None

        for n_workers in WORKER_COUNTS:
            run_idx += 1
            label = f"{n_videos} videos × {n_workers} workers"
            update_state(current_run=label, run_index=run_idx, results=all_results)

            print(f"\n{'─'*55}")
            print(f"  RUN {run_idx}/{total_runs} │ {label}")
            print(f"{'─'*55}")

            cpu_before = psutil.cpu_percent(interval=0.5)
            with tqdm(total=n_videos, desc=f"  Stage2 ({n_workers}w)",
                      unit="vid", ncols=65, colour="green") as pbar:
                timings = run_pipeline(
                    n_workers, wav_subset, creator_of,
                    RESULTS_DIR, gpu_actor, pbar
                )

            cpu_after  = psutil.cpu_percent(interval=0.5)
            avg_cpu    = round((cpu_before + cpu_after) / 2, 1)

            if baseline_time is None:
                baseline_time = timings["total_wall_s"]
            speedup = round(baseline_time / timings["total_wall_s"], 2)

            row = {
                "cpu_workers":    n_workers,
                "videos":         n_videos,
                "stage2_avg_s":   timings["stage2_avg_s"],
                "stage3_avg_s":   timings["stage3_avg_s"],
                "total_wall_s":   timings["total_wall_s"],
                "speedup":        speedup,
                "avg_cpu_pct":    avg_cpu,
            }
            all_results.append(row)
            update_state(results=all_results, stage="idle")

            print(f"  ✅  Stage2 {timings['stage2_avg_s']}s/video | "
                  f"Stage3 {timings['stage3_avg_s']}s/creator | "
                  f"Total {timings['total_wall_s']}s | ×{speedup}")

    # ── Final table ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    print(f"\n\n{'='*70}")
    print("  SCALABILITY MATRIX — Ray Distributed Pipeline")
    print(f"  (Stage2=Whisper/CPU  Stage3=CLIP/GPU  1×RTX GPU + {os.cpu_count()} CPU cores)")
    print(f"{'='*70}")
    for n_vid in VIDEO_COUNTS:
        print(f"\n  Videos = {n_vid}")
        sub = df[df["videos"]==n_vid][
            ["cpu_workers","stage2_avg_s","stage3_avg_s","total_wall_s","speedup","avg_cpu_pct"]
        ]
        print(sub.to_string(index=False))
    print(f"\n{'='*70}")

    df.to_csv("scalability_results.csv", index=False)
    print("💾 Saved → scalability_results.csv")
    update_state(done=True, current_run="Complete", run_index=total_runs)

    print("\n🌐 Dashboard still live at http://localhost:8888 — press Ctrl+C to exit")
    ray.shutdown()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
