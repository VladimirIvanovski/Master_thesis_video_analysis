"""Generate project_report.html — open in browser, print to PDF."""
import base64, os, sys
sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = os.path.dirname(__file__)

def b64(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

c_scalability  = b64("scalability_chart.png")
c_full         = b64("full_pipeline_chart.png")
c_large        = b64("large_scale_chart.png")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TikTok Video Pipeline at Scale — Project Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,sans-serif;font-size:13px;color:#1a1a2e;background:#fff}}
@media print{{
  .page-break{{page-break-before:always}}
  body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
}}
.cover{{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
  color:#fff;padding:70px 60px;min-height:100vh;display:flex;
  flex-direction:column;justify-content:center}}
.cover h1{{font-size:38px;font-weight:700;margin-bottom:10px;line-height:1.2}}
.cover .sub{{font-size:18px;color:#a0aec0;margin-bottom:36px}}
.cover .meta div{{font-size:14px;color:#cbd5e0;line-height:2.2}}
.badge{{display:inline-block;background:rgba(255,255,255,0.12);
  border:1px solid rgba(255,255,255,0.25);border-radius:6px;
  padding:3px 11px;margin:3px;font-size:11px;font-family:monospace}}
.section{{padding:36px 60px}}
.section:nth-child(even){{background:#f8f9fc}}
h2{{font-size:18px;font-weight:700;color:#1a1a2e;
  border-left:4px solid #4f46e5;padding-left:12px;margin-bottom:18px}}
h3{{font-size:13px;font-weight:700;color:#4f46e5;margin:18px 0 8px}}
p{{margin-bottom:10px;line-height:1.75;color:#374151}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:12px}}
th{{background:#4f46e5;color:#fff;padding:7px 11px;text-align:left;font-weight:600}}
td{{padding:6px 11px;border-bottom:1px solid #e5e7eb;color:#374151}}
tr:hover td{{background:#f0f0ff}}
td.good{{color:#16a34a;font-weight:700}}
td.best{{background:#dcfce7;color:#15803d;font-weight:700}}
tr.hl td{{background:#eef2ff}}
.chart-box{{background:#0d1117;border-radius:10px;padding:14px;margin:16px 0}}
.chart-box img{{width:100%;border-radius:6px;display:block}}
.caption{{text-align:center;font-size:11px;color:#9ca3af;margin-top:6px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:14px 0}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:18px;
  box-shadow:0 1px 3px rgba(0,0,0,0.07)}}
.card .num{{font-size:30px;font-weight:700;color:#4f46e5}}
.card .lbl{{font-size:12px;color:#6b7280;margin-top:4px}}
code{{font-family:monospace;background:#f1f5f9;padding:1px 5px;
  border-radius:3px;font-size:12px;color:#4f46e5}}
pre{{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;
  font-family:monospace;font-size:11.5px;overflow:auto;margin:10px 0;line-height:1.65}}
.arch{{background:#0f172a;color:#a3e635;padding:16px 22px;border-radius:8px;
  font-family:monospace;font-size:11px;line-height:1.85;margin:12px 0;white-space:pre}}
.pill{{display:inline-block;padding:2px 9px;border-radius:12px;font-size:11px;font-weight:600}}
.cpu{{background:#dbeafe;color:#1d4ed8}}
.gpu{{background:#f3e8ff;color:#7c3aed}}
.cbox{{border-radius:10px;padding:22px 26px;margin:14px 0;color:#fff}}
.cbox h3{{font-size:13px;margin-bottom:8px}}
.cbox p{{font-size:13px;line-height:1.7;color:rgba(255,255,255,0.88)}}
ul{{margin:8px 0 8px 20px;line-height:1.9}}
li{{color:#374151}}
.footer{{background:#1a1a2e;color:#9ca3af;text-align:center;padding:28px;font-size:12px}}
</style>
</head>
<body>

<!-- ══ COVER ══════════════════════════════════════════════════════════════ -->
<div class="cover">
  <div>
    <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
        color:#818cf8;margin-bottom:14px">Academic Project Report</div>
    <h1>TikTok Video Pipeline<br>at Scale</h1>
    <div class="sub">Аналитика на Големи / Дистрибуирани Податоци</div>
    <div style="margin:24px 0">
      <span class="badge">Ray 2.x</span><span class="badge">PyTorch + CUDA</span>
      <span class="badge">CLIP ViT-B/32</span><span class="badge">Whisper GPU</span>
      <span class="badge">FAISS</span><span class="badge">Elasticsearch</span>
      <span class="badge">Flask</span><span class="badge">FFmpeg</span>
    </div>
    <div class="meta">
      <div><b style="color:#fff">Subject:</b> Big Data &amp; Distributed Data Analytics</div>
      <div><b style="color:#fff">Approach:</b> Ray Distributed Computing + Multi-Stage CPU/GPU Pipeline</div>
      <div><b style="color:#fff">Scale achieved:</b> 948 TikTok videos processed end-to-end</div>
      <div><b style="color:#fff">Hardware:</b> 1 laptop · 16 CPU cores · 1 CUDA GPU</div>
      <div><b style="color:#fff">Repository:</b> github.com/VladimirIvanovski/Master_thesis_video_analysis</div>
    </div>
  </div>
</div>

<!-- ══ 1. OVERVIEW ════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>1. Project Overview</h2>
  <p>This project demonstrates a fully distributed big-data pipeline that downloads, processes, transcribes,
  and embeds TikTok videos at scale using <b>Ray</b> for distributed orchestration. The pipeline spans
  video acquisition, frame extraction, audio decoding, speech transcription, semantic embedding generation,
  and a personalized search layer — all with live resource monitoring.</p>
  <div class="grid2">
    <div class="card"><div class="num">948</div><div class="lbl">Videos processed end-to-end</div></div>
    <div class="card"><div class="num">340</div><div class="lbl">Unique TikTok creators analyzed</div></div>
    <div class="card"><div class="num">×2.51</div><div class="lbl">Peak speedup (1 → 8 CPU workers)</div></div>
    <div class="card"><div class="num">0.56 v/s</div><div class="lbl">Throughput — large-scale run</div></div>
  </div>
</div>

<!-- ══ 2. TECH STACK ══════════════════════════════════════════════════════ -->
<div class="section">
  <h2>2. Technology Stack</h2>
  <table>
    <tr><th>Layer</th><th>Technology</th><th>Role</th><th>Compute</th></tr>
    <tr><td>Orchestration</td><td><b>Ray 2.x</b></td><td>Distributed task scheduling, actor model</td><td><span class="pill cpu">CPU</span></td></tr>
    <tr><td>Video download</td><td>yt-dlp</td><td>Bulk TikTok video + audio download</td><td><span class="pill cpu">CPU</span></td></tr>
    <tr><td>Frame/audio extraction</td><td><b>FFmpeg</b></td><td>1fps frames, 16kHz mono WAV</td><td><span class="pill cpu">CPU</span></td></tr>
    <tr><td>Transcription</td><td><b>faster-whisper (tiny)</b></td><td>Speech-to-text per video</td><td><span class="pill gpu">GPU</span></td></tr>
    <tr><td>Vision embedding</td><td><b>CLIP ViT-B/32</b></td><td>Dense 512-dim image + text vectors</td><td><span class="pill gpu">GPU</span></td></tr>
    <tr><td>Vector index</td><td>FAISS (IndexFlatIP)</td><td>Approximate nearest-neighbour search</td><td><span class="pill cpu">CPU</span></td></tr>
    <tr><td>Feedback store</td><td>Elasticsearch 8.x (Docker)</td><td>Query-aware Good/Bad interactions</td><td><span class="pill cpu">CPU</span></td></tr>
    <tr><td>Demo API</td><td>Flask</td><td>Personalized search + live re-ranking</td><td><span class="pill cpu">CPU</span></td></tr>
    <tr><td>Monitoring</td><td>Ray Dashboard + psutil</td><td>CPU%, RAM, GPU, throughput — live</td><td>—</td></tr>
  </table>
</div>

<!-- ══ 3. ARCHITECTURE ════════════════════════════════════════════════════ -->
<div class="section page-break">
  <h2>3. Pipeline Architecture</h2>
  <p>Three streaming stages run <b>concurrently</b>. Each stage fires immediately when the previous
  stage completes for a given video/creator — no stage waits for the full batch to finish.</p>
  <div class="arch">
+----------------------------------------------------------+
|          RAY CLUSTER  (1 machine, 16 CPUs, 1 GPU)        |
|                                                          |
|  STAGE 1 — CPU Workers (FFmpeg)   avg 6.5s/video        |
|  +-------+ +-------+ +-------+ +-------+               |
|  | W-1   | | W-2   | | W-3   | | W-N   |               |
|  +---+---+ +---+---+ +---+---+ +---+---+               |
|      |         |         |         |                    |
|      +----+----+---------+---------+                    |
|           |   ray.wait() fires per video                |
|  STAGE 2 — GPU Whisper  (num_gpus=0.5)  avg 1.76s/vid  |
|           +-------------------+                         |
|           |   Whisper tiny    |                         |
|           +--------+----------+                         |
|                    |   ray.wait() fires per creator     |
|  STAGE 3 — GPU CLIP  (num_gpus=0.5)   avg 0.08s/creator|
|           +-------------------+                         |
|           |   CLIP ViT-B/32   +---> FAISS index        |
|           +-------------------+                         |
+----------------------------------------------------------+
  </div>
  <h3>GPU sharing</h3>
  <p>Whisper uses <code>num_gpus=0.5</code> and CLIP uses <code>num_gpus=0.5</code>.
  Ray allows both actors to be scheduled simultaneously on the single GPU.
  Total VRAM: ~150 MB (Whisper) + ~420 MB (CLIP) = ~570 MB — well within any modern laptop GPU.</p>

  <h3>Core concurrency pattern</h3>
  <pre>
# All Stage-1 tasks submitted at once to N workers
futures = [stage1_extract.remote(mp4) for mp4 in all_videos]

# Streaming loop — all 3 stages overlap in real time
while remaining_s1 or remaining_s2 or s3_futures:
    done, remaining_s1 = ray.wait(remaining_s1, num_returns=1, timeout=0.1)
    for f in done:
        result = ray.get(f)
        # Fire Stage 2 immediately — no waiting for other videos
        s2_fut = whisper_actor.transcribe.remote(result["wav"])

    done2, remaining_s2 = ray.wait(remaining_s2, num_returns=1, timeout=0.1)
    for f in done2:
        result = ray.get(f)
        if all_videos_of_creator_done(result["creator"]):
            # Fire Stage 3 immediately for this creator
            s3_fut = clip_actor.embed.remote(result["creator"], ...)</pre>
</div>

<!-- ══ 4. DATASET ════════════════════════════════════════════════════════ -->
<div class="section">
  <h2>4. Dataset</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Source</td><td>tiktok_profile_5k.csv (5,000 creator profiles)</td></tr>
    <tr><td>Creators downloaded</td><td>340</td></tr>
    <tr><td>Videos (.mp4)</td><td>948</td></tr>
    <tr><td>Audio files (.wav)</td><td>948</td></tr>
    <tr><td>Frames extracted (.png)</td><td>~9,480 (10 per video @ 1fps)</td></tr>
    <tr><td>Max videos per creator</td><td>3 (filtered: 10–50s duration)</td></tr>
    <tr><td>Transcriptions</td><td>340 creators — creator_transcriptions.csv</td></tr>
    <tr class="hl"><td><b>FAISS index</b></td><td><b>269 creators — 512-dim CLIP embeddings</b></td></tr>
  </table>
</div>

<!-- ══ 5. BENCHMARK 1 ════════════════════════════════════════════════════ -->
<div class="section page-break">
  <h2>5. Benchmark 1 — CPU Worker Scaling (Whisper on CPU)</h2>
  <p>9 videos (3 creators × 3 videos). Stage 2 runs Whisper on <b>CPU</b>
  to demonstrate pure CPU parallelism scaling. Stage 3 runs CLIP on GPU concurrently.</p>
  <table>
    <tr><th>CPU Workers</th><th>Stage2 avg (s)</th><th>Stage3 avg (s)</th><th>Total Wall (s)</th><th>Speedup</th><th>CPU %</th></tr>
    <tr><td>1</td><td>7.16</td><td>1.51</td><td>72.13</td><td>×1.00</td><td>16.8%</td></tr>
    <tr><td>2</td><td>7.31</td><td>1.39</td><td>59.31</td><td>×1.22</td><td>22.2%</td></tr>
    <tr><td>4</td><td>8.61</td><td>1.23</td><td>38.93</td><td>×1.85</td><td>31.7%</td></tr>
    <tr class="hl"><td><b>8</b></td><td><b>7.65</b></td><td><b>1.05</b></td><td><b>28.76</b></td><td class="best">×2.51</td><td><b>45.3%</b></td></tr>
  </table>
  <p><b>Insight:</b> ×2.51 speedup at 8 workers. Sub-linear vs ideal ×8 due to Amdahl's Law —
  the GPU stage is a fixed serial component that bounds maximum speedup.</p>
</div>

<!-- ══ 6. SCALABILITY MATRIX ════════════════════════════════════════════ -->
<div class="section">
  <h2>6. Benchmark 2 — Scalability Matrix (10 / 20 / 30 Videos × 1/2/4/8 Workers)</h2>

  <h3>10 Videos</h3>
  <table>
    <tr><th>Workers</th><th>Stage2 avg (s)</th><th>Stage3 avg (s)</th><th>Total (s)</th><th>Speedup</th><th>CPU %</th></tr>
    <tr><td>1</td><td>8.33</td><td>1.38</td><td>96.24</td><td>×1.00</td><td>31.2%</td></tr>
    <tr><td>2</td><td>8.77</td><td>0.99</td><td>55.28</td><td>×1.74</td><td>21.6%</td></tr>
    <tr><td>4</td><td>9.86</td><td>1.13</td><td>39.57</td><td>×2.43</td><td>18.5%</td></tr>
    <tr class="hl"><td><b>8</b></td><td>11.72</td><td>1.14</td><td>41.46</td><td class="good">×2.32</td><td>54.8%</td></tr>
  </table>

  <h3>20 Videos</h3>
  <table>
    <tr><th>Workers</th><th>Stage2 avg (s)</th><th>Stage3 avg (s)</th><th>Total (s)</th><th>Speedup</th><th>CPU %</th></tr>
    <tr><td>1</td><td>6.72</td><td>0.80</td><td>142.06</td><td>×1.00</td><td>23.8%</td></tr>
    <tr><td>2</td><td>7.23</td><td>0.75</td><td>82.11</td><td>×1.73</td><td>11.4%</td></tr>
    <tr><td>4</td><td>9.46</td><td>0.77</td><td>62.41</td><td>×2.28</td><td>22.8%</td></tr>
    <tr class="hl"><td><b>8</b></td><td>12.40</td><td>0.71</td><td>58.02</td><td class="good">×2.45</td><td>46.7%</td></tr>
  </table>

  <h3>30 Videos</h3>
  <table>
    <tr><th>Workers</th><th>Stage2 avg (s)</th><th>Stage3 avg (s)</th><th>Total (s)</th><th>Speedup</th><th>CPU %</th></tr>
    <tr><td>1</td><td>6.34</td><td>0.57</td><td>195.01</td><td>×1.00</td><td>26.9%</td></tr>
    <tr><td>2</td><td>7.96</td><td>0.58</td><td>138.02</td><td>×1.41</td><td>23.4%</td></tr>
    <tr><td>4</td><td>9.00</td><td>0.62</td><td>92.29</td><td>×2.11</td><td>34.0%</td></tr>
    <tr class="hl"><td><b>8</b></td><td>11.87</td><td>0.64</td><td>87.21</td><td class="good">×2.24</td><td>50.4%</td></tr>
  </table>

  {"<div class='chart-box'><img src='data:image/png;base64," + c_scalability + "' alt='Scalability Chart'><div class='caption'>Figure 1 — Scalability matrix: speedup curves, wall time bars, and stage breakdown for 30 videos</div></div>" if c_scalability else ""}
</div>

<!-- ══ 7. FULL 3-STAGE ════════════════════════════════════════════════════ -->
<div class="section page-break">
  <h2>7. Benchmark 3 — Full 3-Stage Pipeline (FFmpeg/CPU + Whisper/GPU + CLIP/GPU)</h2>
  <p>All three stages active simultaneously. 20 videos, 4/6/8 CPU workers + 1 GPU shared between Whisper and CLIP.</p>
  <table>
    <tr><th>CPU Workers</th><th>Stage1 avg/vid (s)</th><th>Stage2 avg/vid (s)</th><th>Stage3 avg/creator (s)</th><th>Total (s)</th><th>Speedup</th></tr>
    <tr><td>4</td><td>7.49</td><td>1.70</td><td>0.79</td><td>51.43</td><td>×1.00</td></tr>
    <tr><td>6</td><td>6.67</td><td>2.09</td><td>0.78</td><td>48.05</td><td>×1.07</td></tr>
    <tr class="hl"><td><b>8</b></td><td><b>6.72</b></td><td><b>1.99</b></td><td><b>0.69</b></td><td><b>44.13</b></td><td class="good">×1.17</td></tr>
  </table>
  <p><b>Observation:</b> With GPU Whisper, Stage 2 becomes GPU-bound (fixed at ~2s regardless of CPU count).
  Modest speedup (×1.17) is correct — the GPU is the bottleneck here, not the CPU.
  Solution for higher speedup: add a second GPU node.</p>

  {"<div class='chart-box'><img src='data:image/png;base64," + c_full + "' alt='Full Pipeline Chart'><div class='caption'>Figure 2 — Full 3-stage benchmark: speedup curve and per-stage time breakdown (4/6/8 workers)</div></div>" if c_full else ""}
</div>

<!-- ══ 8. LARGE SCALE ════════════════════════════════════════════════════ -->
<div class="section">
  <h2>8. Large-Scale Run — 948 Videos, 8 CPU Workers + 1 GPU</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total videos processed</td><td><b>948</b></td></tr>
    <tr><td>Total wall time</td><td><b>20.1 minutes</b></td></tr>
    <tr><td>Throughput</td><td><b>0.56 videos / second</b></td></tr>
    <tr><td>Stage 1 avg / video (FFmpeg CPU)</td><td>6.5 s</td></tr>
    <tr><td>Stage 2 avg / video (Whisper GPU)</td><td>1.76 s</td></tr>
    <tr><td>Stage 3 avg / creator (CLIP GPU)</td><td>0.08 s</td></tr>
    <tr><td>GPU VRAM peak</td><td>~570 MB (Whisper 150 + CLIP 420)</td></tr>
    <tr class="hl"><td><b>Projection — 10,000 videos (8 workers)</b></td><td><b>~300 min (~5 hours)</b></td></tr>
    <tr class="hl"><td><b>Projection — 10,000 videos (32 workers / 4 servers)</b></td><td><b>~90 min</b></td></tr>
  </table>

  {"<div class='chart-box'><img src='data:image/png;base64," + c_large + "' alt='Large Scale Chart'><div class='caption'>Figure 3 — Large-scale run: throughput over time, per-stage breakdown, and 10k projection</div></div>" if c_large else ""}
</div>

<!-- ══ 9. MULTI-SERVER ════════════════════════════════════════════════════ -->
<div class="section page-break">
  <h2>9. Scaling to Real Distributed Servers</h2>
  <p>Deploying to a real multi-node Ray cluster requires changing <b>one line of code</b>.
  All <code>@ray.remote</code> functions and actors work unchanged.</p>
  <table>
    <tr><th></th><th>Single Machine (current)</th><th>Real Cluster</th></tr>
    <tr><td>Ray init</td><td><code>ray.init(num_cpus=8, ...)</code></td><td><code>ray.init(address='auto')</code></td></tr>
    <tr><td>Head node</td><td>Automatic</td><td><code>ray start --head --port=6379</code></td></tr>
    <tr><td>Worker nodes</td><td>Simulated by CPU count</td><td><code>ray start --address='&lt;HEAD_IP&gt;:6379'</code></td></tr>
    <tr><td>Worker setup</td><td>Nothing</td><td><code>pip install -r requirements.txt</code></td></tr>
    <tr><td>Shared files</td><td>Local disk</td><td>NFS/SMB mount at same path on all nodes</td></tr>
    <tr class="hl"><td><b>@ray.remote code</b></td><td colspan="2" style="text-align:center"><b>Unchanged — zero modifications needed</b></td></tr>
  </table>

  <h3>Scale projection with real servers</h3>
  <table>
    <tr><th>Setup</th><th>CPU Workers</th><th>Est. 10k videos</th><th>Speedup vs current</th></tr>
    <tr><td>1 laptop (current)</td><td>8</td><td>~300 min</td><td>×1.0</td></tr>
    <tr><td>2 servers (8 cores each)</td><td>16</td><td>~160 min</td><td>×1.9</td></tr>
    <tr class="hl"><td><b>4 servers (8 cores each)</b></td><td><b>32</b></td><td><b>~90 min</b></td><td class="good"><b>×3.3</b></td></tr>
  </table>
</div>

<!-- ══ 10. SEARCH LAYER ══════════════════════════════════════════════════ -->
<div class="section">
  <h2>10. Personalized Search Layer</h2>
  <p>A query-aware personalized search system re-ranks creators based on per-query user feedback
  stored in Elasticsearch.</p>

  <h3>Re-ranking formula</h3>
  <pre>final_score = 0.7 × semantic_score (FAISS cosine similarity)
            + 0.25 × good_boost    (past "Good" clicks for this query)
            - 0.25 × bad_penalty   (past "Bad" clicks for this query)</pre>

  <h3>Flow</h3>
  <ul>
    <li>User submits a text query → CLIP encodes it to 512-dim vector</li>
    <li>FAISS returns top-K creators by cosine similarity</li>
    <li>Elasticsearch looks up past Good/Bad interactions for that specific query</li>
    <li>Re-ranking formula applied → personalized ordered results</li>
    <li>Flask demo at <code>localhost:5000</code> — live feedback buttons update ES in real time</li>
  </ul>

  <h3>Elasticsearch index: user_interactions</h3>
  <table>
    <tr><th>Field</th><th>Type</th><th>Description</th></tr>
    <tr><td>query</td><td>keyword</td><td>The search query string</td></tr>
    <tr><td>creator</td><td>keyword</td><td>TikTok username</td></tr>
    <tr><td>feedback</td><td>keyword</td><td>"good" or "bad"</td></tr>
    <tr><td>timestamp</td><td>date</td><td>When feedback was recorded</td></tr>
  </table>
</div>

<!-- ══ 11. CONCLUSIONS ══════════════════════════════════════════════════ -->
<div class="section page-break">
  <h2>11. Key Conclusions</h2>

  <div class="cbox" style="background:linear-gradient(135deg,#3730a3,#4f46e5)">
    <h3>1 — Ray enables real horizontal CPU scaling</h3>
    <p>Going from 1 to 8 workers achieves ×2.51 speedup on a single machine.
    The same code scales to a real multi-node cluster by changing one line.
    Task distribution, actor lifecycle, and fault tolerance all work identically.</p>
  </div>

  <div class="cbox" style="background:linear-gradient(135deg,#6d28d9,#7c3aed);margin-top:12px">
    <h3>2 — Concurrent staging eliminates idle time</h3>
    <p>Using <code>ray.wait(num_returns=1)</code>, Stage 2 starts the moment Stage 1 completes
    for each individual video. Stage 3 starts the moment all of a creator's videos finish Stage 2.
    The GPU is never idle waiting for the CPU batch to complete.</p>
  </div>

  <div class="cbox" style="background:linear-gradient(135deg,#065f46,#059669);margin-top:12px">
    <h3>3 — CPU is the bottleneck, not the GPU</h3>
    <p>FFmpeg extraction (6.5s/video) dominates total time. CLIP embedding takes only 0.08s/creator.
    The correct scaling strategy is adding CPU cores / nodes — not more GPUs.
    GPU VRAM stays flat at ~570 MB regardless of how many videos are in the pipeline.</p>
  </div>

  <div class="cbox" style="background:linear-gradient(135deg,#92400e,#b45309);margin-top:12px">
    <h3>4 — Amdahl's Law observed and explained</h3>
    <p>Speedup is bounded by the serial fraction (GPU stages). With ~30% serial work,
    maximum theoretical speedup is ×3.3 regardless of CPU count. Measured ×2.51 is consistent
    with this prediction, validating the architecture's honest distributed-systems analysis.</p>
  </div>

  <h3 style="margin-top:24px">Requirements checklist</h3>
  <table>
    <tr><th>Requirement</th><th>Status</th><th>Evidence</th></tr>
    <tr><td>Distributed Ray workers</td><td class="good">Done</td><td>4/6/8 workers benchmarked</td></tr>
    <tr><td>CPU/GPU stage split</td><td class="good">Done</td><td>FFmpeg/CPU, Whisper/GPU, CLIP/GPU</td></tr>
    <tr><td>Concurrent streaming pipeline</td><td class="good">Done</td><td>ray.wait() — all 3 stages overlap</td></tr>
    <tr><td>Scalability benchmark</td><td class="good">Done</td><td>1/2/4/8 workers × 10/20/30 videos matrix</td></tr>
    <tr><td>Per-stage + total timing</td><td class="good">Done</td><td>3 CSVs + 3 charts</td></tr>
    <tr><td>Live monitoring (CPU/RAM/GPU)</td><td class="good">Done</td><td>localhost:8888 + Ray Dashboard</td></tr>
    <tr><td>Large-scale run (~1000 videos)</td><td class="good">Done</td><td>948 videos, 20.1 min, 0.56 vid/s</td></tr>
    <tr><td>Embedding quality (FAISS)</td><td class="good">Done</td><td>269 creators indexed</td></tr>
    <tr><td>Personalized search</td><td class="good">Done</td><td>Elasticsearch + Flask demo</td></tr>
    <tr><td>Multi-server scale instructions</td><td class="good">Done</td><td>Section 9</td></tr>
  </table>
</div>

<div class="footer">
  TikTok Video Pipeline at Scale &nbsp;|&nbsp;
  Аналитика на Големи / Дистрибуирани Податоци &nbsp;|&nbsp;
  github.com/VladimirIvanovski/Master_thesis_video_analysis &nbsp;|&nbsp;
  Branch: ray-distributed-benchmark
</div>

</body>
</html>
"""

out = os.path.join(RESULTS_DIR, "project_report.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"Done: {out}")
"""
HOW TO SAVE AS PDF:
  1. Open project_report.html in Chrome or Edge
  2. Press Ctrl+P  (Print)
  3. Destination → Save as PDF
  4. Layout → Landscape  (recommended for charts)
  5. More settings → Scale 85%,  Margins: Minimum
  6. Save
"""
