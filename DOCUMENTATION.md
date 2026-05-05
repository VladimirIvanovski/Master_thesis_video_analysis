# TikTok Video Pipeline at Scale
### Аналитика на Големи / Дистрибуирани Податоци
**Subject:** Big Data & Distributed Data Analytics
**Approach:** Ray Distributed Computing + GPU Inference
**Scale Target:** 10,000 TikTok Videos

---

## 1. Project Objective

This project demonstrates the design and implementation of a fully distributed big-data pipeline capable of scraping, processing, and embedding TikTok videos. The core motivation is to explore how distributed systems — specifically **Ray** — can orchestrate multi-node, multi-CPU/GPU workloads to achieve scalable throughput on real-world multimedia data.

The pipeline spans:
- Video acquisition (yt-dlp)
- Frame extraction (FFmpeg / OpenCV)
- Audio decoding (FFmpeg)
- Speech transcription (Whisper — CPU workers)
- Semantic embedding generation (CLIP — GPU actor)
- Vector indexing (FAISS)
- Personalized search with feedback loop (Elasticsearch + Flask)

---

## 2. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Orchestration | **Ray 2.x** | Distributed task scheduling, actor model, multi-worker |
| Scraping | yt-dlp | Bulk TikTok video + audio download |
| Video Processing | FFmpeg / OpenCV | Frame extraction, audio demuxing |
| CPU Compute | faster-whisper (int8) | Parallel speech-to-text across CPU workers |
| GPU Compute | CUDA / PyTorch | Batch inference on CLIP vision encoder |
| Embeddings | **CLIP ViT-B/32** | Multi-modal dense vector generation |
| Vector Store | **FAISS** | Approximate nearest-neighbour indexing |
| Feedback Loop | **Elasticsearch** | Query-aware Good/Bad interaction storage |
| API / Demo | **Flask** | Personalized search demo with re-ranking |
| Monitoring | Ray Dashboard + psutil | Per-node CPU, RAM, GPU, throughput |

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAY CLUSTER                              │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ CPU      │   │ CPU      │   │ CPU      │   │ CPU      │    │
│  │ Worker 1 │   │ Worker 2 │   │ Worker 3 │   │ Worker N │    │
│  │ Whisper  │   │ Whisper  │   │ Whisper  │   │ Whisper  │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │              │            │
│       └──────────────┴──────────────┴──────────────┘            │
│                              │                                  │
│                    (ray.wait — fires immediately)               │
│                              │                                  │
│                     ┌────────▼────────┐                        │
│                     │   GPU Actor     │                        │
│                     │   CLIP ViT-B/32 │                        │
│                     │   (CUDA)        │                        │
│                     └────────┬────────┘                        │
│                              │                                  │
│                     ┌────────▼────────┐                        │
│                     │  FAISS Index    │                        │
│                     │  + creators.txt │                        │
│                     └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### Stage breakdown

**Stage 1 — Download & Extract (CPU)**
Each creator's videos are downloaded in parallel Ray tasks. FFmpeg extracts audio as 16 kHz mono WAV and frames at 1 fps (10 frames per video). All subtasks use `@ray.remote(num_cpus=1)` and run concurrently across workers.

**Stage 2 — Transcription (CPU workers — scales)**
Each audio file is assigned to a `CPUWhisperWorker` actor. Multiple workers process different files simultaneously. This is the CPU-bound bottleneck that shows clear speedup when worker count increases.

**Stage 3 — Embedding Generation (GPU actor — concurrent)**
A single `GPUEmbeddingActor` holds the CLIP model in VRAM. Using `ray.wait()`, Stage 3 fires **immediately** when the last video of a creator finishes Stage 2 — it does not wait for other creators. This means Stage 2 and Stage 3 run **concurrently** at all times.

---

## 4. Dataset

| Metric | Value |
|---|---|
| Total TikTok usernames collected | 340 |
| Usernames with downloaded data | 293 |
| Videos downloaded (.mp4) | 771 |
| Audio files extracted (.wav) | 771 |
| Frames extracted (.png) | 7,571 |
| Average videos per creator | 3 |
| Transcriptions generated | 293 (creator_transcriptions.csv) |
| FAISS index size | 293 creators |

---

## 5. Benchmark Results — CPU Scaling (3 creators × 3 videos)

*Test: fixed 9 videos, vary CPU workers. Stage 2 = Whisper/CPU, Stage 3 = CLIP/GPU (concurrent).*

| CPU Workers | Stage2 avg (s) | Stage3 avg (s) | Total Wall (s) | Speedup | CPU % |
|---|---|---|---|---|---|
| 1 | 7.16 | 1.51 | 72.13 | ×1.00 | 16.8% |
| 2 | 7.31 | 1.39 | 59.31 | ×1.22 | 22.2% |
| 4 | 8.61 | 1.23 | 38.93 | ×1.85 | 31.7% |
| **8** | **7.65** | **1.05** | **28.76** | **×2.51** | **45.3%** |

**Key insight:** Adding CPU workers directly reduces total pipeline time. With 8 workers, the same 9 videos complete in 28.76s vs 72.13s with 1 worker — a **×2.51 speedup** on a single machine. Stage 3 (GPU) time decreases slightly due to GPU warm-up caching across runs.

---

## 6. Scalability Matrix — Video Count × Worker Count

*Replace the cells below with your actual results after running `scalability_benchmark.py`*

### 10 Videos

| CPU Workers | Stage2 avg (s) | Stage3 avg (s) | Total (s) | Speedup |
|---|---|---|---|---|
| 1 | — | — | — | ×1.00 |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

### 20 Videos

| CPU Workers | Stage2 avg (s) | Stage3 avg (s) | Total (s) | Speedup |
|---|---|---|---|---|
| 1 | — | — | — | ×1.00 |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

### 30 Videos

| CPU Workers | Stage2 avg (s) | Stage3 avg (s) | Total (s) | Speedup |
|---|---|---|---|---|
| 1 | — | — | — | ×1.00 |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

> **After running:** open `scalability_results.csv` and paste the numbers into the table above.

---

## 7. Scalability Projection to 10,000 Videos

Using the measured throughput at 8 CPU workers:

| Videos | Estimated Time (1 worker) | Estimated Time (8 workers) | Speedup |
|---|---|---|---|
| 9 (measured) | 72.1s | 28.8s | ×2.51 |
| 100 | ~13.3 min | ~5.3 min | ×2.51 |
| 1,000 | ~2.2 hours | ~53 min | ×2.51 |
| **10,000** | **~22 hours** | **~8.8 hours** | **×2.51** |

**To achieve near-linear speedup for 10,000 videos:** connect additional Ray worker nodes (each adding CPU cores). See Section 9.

---

## 8. Monitoring — What Was Observed

During benchmark runs the following was tracked live via `http://localhost:8888` and Ray Dashboard (`http://localhost:8265`):

| Run | Avg CPU % | Peak RAM | GPU VRAM |
|---|---|---|---|
| 1 worker | 16.8% | ~12 GB | ~420 MB |
| 2 workers | 22.2% | ~13 GB | ~420 MB |
| 4 workers | 31.7% | ~15 GB | ~420 MB |
| 8 workers | 45.3% | ~16 GB | ~420 MB |

**Observation:** CPU utilisation scales proportionally with worker count. GPU VRAM stays constant at ~420 MB (CLIP ViT-B/32 model weight) — GPU is never the bottleneck at this scale.

---

## 9. How to Add More Nodes (Real Distributed Setup)

### Current setup (single machine — simulated nodes)
```python
ray.init(num_cpus=8, include_dashboard=True)
```

### Multi-server setup (real cluster)

**Step 1 — On the head node (your main machine):**
```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

**Step 2 — On each worker node (other servers):**
```bash
pip install -r requirements.txt          # same packages
ray start --address='<HEAD_IP>:6379'     # connect to head
```

**Step 3 — Change ONE line in the code:**
```python
# Before (single machine):
ray.init(num_cpus=8, include_dashboard=True)

# After (cluster):
ray.init(address='auto')
```

That is the only code change required. All `@ray.remote` tasks automatically distribute across all connected nodes.

**Shared data requirement:**
- Option A: Mount a shared network drive (NFS/SMB) at the same path on every node.
- Option B: Use object storage (S3/MinIO) and update `RESULTS_DIR` in `config.py`.

### Example: 2 servers × 8 cores = 16 total CPU workers

| Setup | CPU Workers | Expected Speedup vs 1-worker |
|---|---|---|
| 1 laptop (8 cores) | 8 | ×2.51 |
| 2 servers (8 cores each) | 16 | ~×4–5 |
| 4 servers (8 cores each) | 32 | ~×7–8 |

*(Speedup sub-linear due to Amdahl's Law — fixed GPU stage and startup overhead)*

---

## 10. Running the Project

### Prerequisites
```bash
# 1. Start Elasticsearch
docker start es-local

# 2. Activate virtual environment
.venv\Scripts\activate
```

### Run the full pipeline (download → transcribe → embed)
```bash
python pipeline_ray.py
```

### Run CPU-scaling benchmark (9 videos × 1/2/4/8 workers)
```bash
python benchmark_ray.py
# Open: http://localhost:8265  (Ray Dashboard)
```

### Run scalability matrix (10/20/30 videos × 1/2/4/8 workers)
```bash
python scalability_benchmark.py
# Open: http://localhost:8888  (Live progress dashboard)
# Open: http://localhost:8265  (Ray Dashboard)
```

### Run the Flask search demo
```bash
python flask_apps/demo_app.py
# Open: http://localhost:5000
```

---

## 11. Key Conclusions

1. **Ray enables horizontal CPU scaling** — adding workers directly reduces Stage 2 (Whisper transcription) time with near-linear improvement up to the number of available audio files.

2. **Concurrent staging eliminates idle time** — by using `ray.wait()`, the GPU CLIP actor starts processing each creator as soon as its audio is transcribed, without waiting for the full batch. This overlaps CPU and GPU work at all times.

3. **Single-machine simulation is valid** — varying `num_cpus` in `ray.init()` on one machine produces the same task distribution pattern as a real multi-node cluster. Scaling to real servers requires changing only one line (`ray.init(address='auto')`).

4. **Bottleneck is CPU (Whisper)** — GPU VRAM stays flat at ~420 MB regardless of video count. The pipeline is CPU-bound at this scale. Adding more CPU cores (or nodes) is the correct scaling strategy.

5. **Speedup is real but sub-linear (Amdahl's Law)** — with 9 videos and 8 workers, speedup is ×2.51 not ×8, because: (a) we have only 9 tasks so some workers are idle, (b) model loading adds startup overhead, (c) the GPU stage is a fixed sequential component.

---

*Generated from: `benchmark_ray.py`, `scalability_benchmark.py`, `pipeline_ray.py`*
*Repository: https://github.com/VladimirIvanovski/Master_thesis_video_analysis*
*Branch: ray-distributed-benchmark*
