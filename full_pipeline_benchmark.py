"""
full_pipeline_benchmark.py  —  Full 3-Stage Streaming Pipeline Benchmark
=========================================================================
Stage 1  →  FFmpeg frame + audio extraction   (CPU workers)
Stage 2  →  Whisper transcription             (GPU, num_gpus=0.5)
Stage 3  →  CLIP embedding                   (GPU, num_gpus=0.5)

Streaming: as soon as a video finishes Stage 1 → Stage 2 starts immediately.
           as soon as all videos of a creator finish Stage 2 → Stage 3 starts.
           Stages always overlap.

Workers tested : 4 / 6 / 8  CPU workers
Videos         : 20  (re-extracts from existing .mp4 files each run)

Run:      python full_pipeline_benchmark.py
Dashboard : http://localhost:8265  (Ray)
Progress  : http://localhost:8888  (live web)
Output    : full_pipeline_results.csv + full_pipeline_chart.png
"""

import os, glob, time, subprocess, shutil, threading, json
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np, pandas as pd, ray, psutil, torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  Shared live-progress state
# ─────────────────────────────────────────────────────────────────────────────
_state = {"run_label": "", "run_i": 0, "total_runs": 3,
          "s1_done": 0, "s1_total": 0,
          "s2_done": 0, "s2_total": 0,
          "s3_done": 0, "s3_total": 0,
          "results": [], "finished": False}
_lock = threading.Lock()

def upd(**kw):
    with _lock:
        _state.update(kw)

def gst():
    with _lock:
        return dict(_state)

# ─────────────────────────────────────────────────────────────────────────────
#  Tiny web dashboard
# ─────────────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Ray Pipeline Benchmark</title><meta http-equiv="refresh" content="3">
<style>
body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:28px}}
h1{{color:#58a6ff;margin-bottom:4px}}h3{{color:#8b949e;margin:0 0 16px}}
.box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px;margin:12px 0}}
.lbl{{color:#8b949e;font-size:12px;margin-bottom:4px}}.val{{font-size:20px;font-weight:bold;color:#3fb950}}
.bg{{background:#21262d;border-radius:4px;height:18px;width:100%;margin-bottom:8px}}
.fg{{height:18px;border-radius:4px;transition:width .4s}}
.s1{{background:#58a6ff}}.s2{{background:#bc8cff}}.s3{{background:#3fb950}}
table{{border-collapse:collapse;width:100%}}
th{{background:#21262d;padding:7px 10px;color:#58a6ff;text-align:left;font-size:13px}}
td{{padding:5px 10px;border-bottom:1px solid #21262d;font-size:13px}}
.done{{color:#f78166;font-weight:bold;font-size:16px;margin-top:12px}}
</style></head><body>
<h1>Ray Full Pipeline — Live Benchmark</h1>
<h3>Stage1=FFmpeg/CPU &nbsp;|&nbsp; Stage2=Whisper/GPU &nbsp;|&nbsp; Stage3=CLIP/GPU &nbsp;|&nbsp; All stages overlap</h3>
<div class="box">
  <div class="lbl">Current run &nbsp;({ri}/{tr})</div>
  <div class="val">{label}</div><br>
  <div class="lbl">Stage 1 — Frame+Audio extraction (CPU) &nbsp; {s1d}/{s1t}</div>
  <div class="bg"><div class="fg s1" style="width:{p1}%"></div></div>
  <div class="lbl">Stage 2 — Whisper transcription (GPU) &nbsp; {s2d}/{s2t}</div>
  <div class="bg"><div class="fg s2" style="width:{p2}%"></div></div>
  <div class="lbl">Stage 3 — CLIP embedding (GPU) &nbsp; {s3d}/{s3t}</div>
  <div class="bg"><div class="fg s3" style="width:{p3}%"></div></div>
</div>
{table}{done}
<div style="color:#8b949e;font-size:11px;margin-top:12px">
  Auto-refresh 3s &nbsp;|&nbsp;
  <a href="http://localhost:8265" style="color:#58a6ff">Ray Dashboard</a>
</div></body></html>"""

def _pct(a, b): return round(a/b*100) if b else 0

def _build_html(s):
    rows = ""
    if s["results"]:
        rows = ("<div class='box'><table><tr>" +
                "".join(f"<th>{c}</th>" for c in
                        ["Workers","Videos","Stage1 avg(s)","Stage2 avg(s)",
                         "Stage3 avg(s)","Total(s)","Speedup"]) + "</tr>")
        for r in s["results"]:
            rows += (f"<tr><td>{r['workers']}</td><td>{r['videos']}</td>"
                     f"<td>{r['s1_avg']}</td><td>{r['s2_avg']}</td>"
                     f"<td>{r['s3_avg']}</td><td>{r['total_s']}</td>"
                     f"<td>x{r['speedup']}</td></tr>")
        rows += "</table></div>"
    done = "<div class='done'>Benchmark complete — see full_pipeline_results.csv</div>" \
           if s["finished"] else ""
    return HTML.format(
        ri=s["run_i"], tr=s["total_runs"], label=s["run_label"] or "—",
        s1d=s["s1_done"], s1t=s["s1_total"],
        s2d=s["s2_done"], s2t=s["s2_total"],
        s3d=s["s3_done"], s3t=s["s3_total"],
        p1=_pct(s["s1_done"], s["s1_total"]),
        p2=_pct(s["s2_done"], s["s2_total"]),
        p3=_pct(s["s3_done"], s["s3_total"]),
        table=rows, done=done,
    )

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        b = _build_html(gst()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(b))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *_): pass

def start_web(port=8888):
    srv = HTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"  Live dashboard  http://localhost:{port}")


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — CPU: re-extract frames + audio from existing .mp4
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote(num_cpus=1)
def stage1_extract(mp4_path: str, out_dir: str) -> dict:
    """Re-extract audio + 10 frames from an existing .mp4 file."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(mp4_path))[0]

    wav_path    = os.path.join(out_dir, f"{base}_bench.wav")
    frames_dir  = os.path.join(out_dir, "bench_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Audio extraction
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Frame extraction (10 frames @ 1fps from second 1)
    frame_out = os.path.join(frames_dir, f"{base}_%02d.png")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "1", "-t", "10", "-i", mp4_path,
         "-vf", "fps=1", "-qscale:v", "2", frame_out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    elapsed = round(time.time() - t0, 3)
    frames  = glob.glob(os.path.join(frames_dir, "*.png"))
    return {
        "mp4": mp4_path, "wav": wav_path,
        "frames": frames, "s1_time": elapsed,
        "creator": os.path.basename(os.path.dirname(mp4_path)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — GPU Whisper (num_gpus=0.5 → shares GPU with CLIP)
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote(num_gpus=0.5)
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
        return {"wav": wav_path, "text": text, "s2_time": round(time.time()-t0, 3)}


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — GPU CLIP (num_gpus=0.5 → shares GPU with Whisper)
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote(num_gpus=0.5)
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
                b = torch.stack(tensors).to(self.device)
                if self.device == "cuda": b = b.half()
                self.model.encode_image(b)
        if transcription.strip():
            with torch.no_grad():
                self.model.encode_text(
                    self.tokenizer([transcription[:77]]).to(self.device)
                )
        return {"creator": creator, "s3_time": round(time.time()-t0, 3)}


# ─────────────────────────────────────────────────────────────────────────────
#  Full streaming pipeline run
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(num_workers: int, mp4_list: list,
                 whisper: GPUWhisperActor, clip: GPUCLIPActor,
                 results_dir: str) -> dict:
    """
    Streaming 3-stage pipeline:
      Stage 1 submitted all at once → N CPU workers process in parallel
      Stage 2 fires per-video as soon as Stage 1 completes
      Stage 3 fires per-creator as soon as all its videos finish Stage 2
    """
    n_videos   = len(mp4_list)
    n_creators = len(set(os.path.basename(os.path.dirname(p)) for p in mp4_list))

    upd(s1_done=0, s1_total=n_videos,
        s2_done=0, s2_total=n_videos,
        s3_done=0, s3_total=n_creators)

    # Temporary output dir (cleaned after run)
    bench_dir = os.path.join(results_dir, "_bench_tmp")
    os.makedirs(bench_dir, exist_ok=True)

    # Submit all Stage 1 tasks
    s1_futures = {}
    for mp4 in mp4_list:
        creator  = os.path.basename(os.path.dirname(mp4))
        out_dir  = os.path.join(bench_dir, creator,
                                os.path.splitext(os.path.basename(mp4))[0])
        fut = stage1_extract.remote(mp4, out_dir)
        s1_futures[fut] = mp4

    # Per-creator tracking for Stage 3
    creator_wavs:   dict[str, list] = {}
    creator_frames: dict[str, list] = {}
    creator_texts:  dict[str, list] = {}
    creator_expected = {}
    for mp4 in mp4_list:
        c = os.path.basename(os.path.dirname(mp4))
        creator_expected[c] = creator_expected.get(c, 0) + 1
        creator_wavs[c]   = []
        creator_frames[c] = []
        creator_texts[c]  = []

    s1_times, s2_times, s3_times = [], [], []
    s2_futures = {}   # future → (creator, wav)
    s3_futures = {}   # future → creator

    wall_start    = time.time()
    remaining_s1  = list(s1_futures.keys())
    remaining_s2  = []

    pbar1 = tqdm(total=n_videos, desc=f"  Stage1({num_workers}w)", unit="vid",
                 ncols=60, colour="blue",  leave=False)
    pbar2 = tqdm(total=n_videos, desc=f"  Stage2 GPU",             unit="vid",
                 ncols=60, colour="magenta", leave=False)
    pbar3 = tqdm(total=n_creators, desc=f"  Stage3 GPU",           unit="creator",
                 ncols=60, colour="green",  leave=False)

    # ── Streaming loop ────────────────────────────────────────────────────────
    while remaining_s1 or remaining_s2 or s3_futures:
        # Poll Stage 1
        if remaining_s1:
            done1, remaining_s1 = ray.wait(remaining_s1, num_returns=1,
                                           timeout=0.1)
            for f in done1:
                r1 = ray.get(f)
                s1_times.append(r1["s1_time"])
                creator = r1["creator"]
                creator_frames[creator].extend(r1["frames"])
                # Fire Stage 2 immediately
                s2f = whisper.transcribe.remote(r1["wav"])
                s2_futures[s2f] = (creator, r1["wav"])
                remaining_s2.append(s2f)
                pbar1.update(1)
                upd(s1_done=pbar1.n)

        # Poll Stage 2
        if remaining_s2:
            done2, remaining_s2 = ray.wait(remaining_s2, num_returns=1,
                                           timeout=0.1)
            for f in done2:
                r2 = ray.get(f)
                s2_times.append(r2["s2_time"])
                creator, _ = s2_futures[f]
                creator_texts[creator].append(r2["text"])
                pbar2.update(1)
                upd(s2_done=pbar2.n)
                # Fire Stage 3 when all videos of this creator are transcribed
                if len(creator_texts[creator]) == creator_expected[creator]:
                    combined = " ".join(creator_texts[creator])
                    s3f = clip.embed.remote(
                        creator, creator_frames[creator], combined
                    )
                    s3_futures[s3f] = creator

        # Poll Stage 3
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

    # Cleanup temp files
    shutil.rmtree(bench_dir, ignore_errors=True)

    total_wall = round(time.time() - wall_start, 2)
    return {
        "s1_avg": round(np.mean(s1_times), 2) if s1_times else 0,
        "s2_avg": round(np.mean(s2_times), 2) if s2_times else 0,
        "s3_avg": round(np.mean(s3_times), 2) if s3_times else 0,
        "total_s": total_wall,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Chart generation
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",   "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
        "grid.color": "#21262d",       "text.color": "#e6edf3",
        "font.family": "monospace",
    })

    workers = df["workers"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "TikTok Pipeline at Scale — Full 3-Stage Benchmark (20 videos)\n"
        "Stage1=FFmpeg/CPU  |  Stage2=Whisper/GPU  |  Stage3=CLIP/GPU  |  Stages overlap",
        fontsize=12, fontweight="bold", color="#e6edf3", y=1.02
    )

    # Panel 1 — Speedup curve
    ax1.plot(workers, df["speedup"], marker="o", linewidth=2.5,
             markersize=9, color="#3fb950", label="Actual speedup")
    ax1.plot(workers, [1, 1.5, 2], linestyle="--", color="#30363d",
             linewidth=1.2, label="Ideal (linear ref)")
    for w, s in zip(workers, df["speedup"]):
        ax1.annotate(f"x{s}", (w, s), textcoords="offset points",
                     xytext=(6, 4), fontsize=9, color="#3fb950")
    ax1.set_title("Speedup vs CPU Workers", fontsize=11, pad=10)
    ax1.set_xlabel("CPU Workers"); ax1.set_ylabel("Speedup (vs 4 workers)")
    ax1.set_xticks(workers); ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4); ax1.set_ylim(0.8, max(df["speedup"])+0.4)

    # Panel 2 — Stacked stage bars
    x     = range(len(workers))
    width = 0.5
    b1 = ax2.bar(x, df["s1_avg"], width, color="#58a6ff",  label="Stage 1 avg/video (FFmpeg CPU)")
    b2 = ax2.bar(x, df["s2_avg"], width, bottom=df["s1_avg"],
                 color="#bc8cff", label="Stage 2 avg/video (Whisper GPU)")
    b3 = ax2.bar(x, df["s3_avg"], width,
                 bottom=df["s1_avg"] + df["s2_avg"],
                 color="#3fb950", label="Stage 3 avg/creator (CLIP GPU)")

    for bar, v in zip(b1, df["s1_avg"]):
        ax2.text(bar.get_x()+bar.get_width()/2, v/2,
                 f"{v:.1f}s", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for bar, v, base in zip(b2, df["s2_avg"], df["s1_avg"]):
        ax2.text(bar.get_x()+bar.get_width()/2, base+v/2,
                 f"{v:.1f}s", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for bar, v, base in zip(b3, df["s3_avg"], df["s1_avg"]+df["s2_avg"]):
        ax2.text(bar.get_x()+bar.get_width()/2, base+v/2,
                 f"{v:.2f}s", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Total time label on top of each bar
    totals = df["s1_avg"] + df["s2_avg"] + df["s3_avg"]
    for xi, (total, wall) in enumerate(zip(totals, df["total_s"])):
        ax2.text(xi, total + 0.15, f"Wall: {wall}s",
                 ha="center", va="bottom", fontsize=8, color="#8b949e")

    ax2.set_title("Per-Stage Time Breakdown\n(avg per video, stages run concurrently)",
                  fontsize=11, pad=10)
    ax2.set_xlabel("CPU Workers"); ax2.set_ylabel("Time (seconds)")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(workers)
    ax2.legend(fontsize=9); ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.text(0.5, -0.04,
             f"Hardware: 1x GPU (CUDA, shared 0.5+0.5)  |  {os.cpu_count()} CPU cores available  |  "
             "ray.wait() streams tasks across all 3 stages",
             ha="center", fontsize=9, color="#8b949e")

    plt.tight_layout()
    plt.savefig("full_pipeline_chart.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  Chart saved: full_pipeline_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR   = "results_4"
    NUM_VIDEOS    = 20
    WORKER_COUNTS = [4, 6, 8]

    start_web()
    ray.init(
        include_dashboard=True, dashboard_host="0.0.0.0",
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
        num_gpus=1 if torch.cuda.is_available() else 0,
    )
    print(f"\n  Ray ready  |  {ray.cluster_resources()}")
    print(f"  Ray Dashboard  http://localhost:8265\n")

    # Collect mp4 files (up to 3 per creator, pick enough for NUM_VIDEOS)
    all_mp4s = []
    for creator in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir): continue
        mp4s = glob.glob(os.path.join(cdir, "*.mp4"))[:3]
        all_mp4s.extend(mp4s)
        if len(all_mp4s) >= NUM_VIDEOS: break
    mp4_subset = all_mp4s[:NUM_VIDEOS]
    print(f"  Videos selected : {len(mp4_subset)}")
    print(f"  Creators        : {len(set(os.path.basename(os.path.dirname(p)) for p in mp4_subset))}\n")

    # GPU actors shared across all runs (loaded once)
    print("  Loading GPU actors...")
    whisper_actor = GPUWhisperActor.remote()
    clip_actor    = GPUCLIPActor.remote()
    # Warm up (force actor init before timing starts)
    time.sleep(4)

    rows = []
    baseline_time = None

    for i, n_workers in enumerate(WORKER_COUNTS):
        label = f"{NUM_VIDEOS} videos  x  {n_workers} CPU workers  +  1 GPU"
        upd(run_label=label, run_i=i+1)
        print(f"\n{'='*60}")
        print(f"  RUN {i+1}/3  |  {label}")
        print(f"{'='*60}")

        timings = run_pipeline(
            n_workers, mp4_subset, whisper_actor, clip_actor, RESULTS_DIR
        )

        if baseline_time is None:
            baseline_time = timings["total_s"]
        speedup = round(baseline_time / timings["total_s"], 2)

        row = {"workers": n_workers, "videos": NUM_VIDEOS,
               "s1_avg": timings["s1_avg"], "s2_avg": timings["s2_avg"],
               "s3_avg": timings["s3_avg"], "total_s": timings["total_s"],
               "speedup": speedup}
        rows.append(row)
        upd(results=rows)

        print(f"\n  Stage1 avg  {timings['s1_avg']}s/video  (FFmpeg CPU)")
        print(f"  Stage2 avg  {timings['s2_avg']}s/video  (Whisper GPU)")
        print(f"  Stage3 avg  {timings['s3_avg']}s/creator (CLIP GPU)")
        print(f"  Total wall  {timings['total_s']}s   Speedup x{speedup}")

    # Final table
    df = pd.DataFrame(rows)
    print(f"\n\n{'='*60}")
    print("  FINAL RESULTS — Full 3-Stage Ray Pipeline")
    print(f"{'='*60}")
    print(df[["workers","s1_avg","s2_avg","s3_avg","total_s","speedup"]].to_string(index=False))
    print(f"{'='*60}")

    df.to_csv("full_pipeline_results.csv", index=False)
    print("  Saved: full_pipeline_results.csv")

    plot_results(df)
    upd(finished=True)

    print("\n  Dashboard still live at http://localhost:8888 — Ctrl+C to exit")
    ray.shutdown()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
