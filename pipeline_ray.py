import os
import time
import ray
import pandas as pd
import numpy as np
from config import *          # must stay at top-level, not inside functions
from utils import ensure_dir
from stage1_download_extract import process_creator
from stage2_transcribe import WhisperActor
from stage3_embeddings import EmbeddingActor
from elasticsearch import Elasticsearch
import psutil
import torch
import threading
# ------------------- Elasticsearch Setup -------------------
es = Elasticsearch("http://localhost:9200")   # Local Elasticsearch
INDEX_NAME = "creator_transcriptions"

def save_to_elasticsearch(username, transcription):
    """Save creator transcription to Elasticsearch."""
    doc = {
        "creator": username,
        "transcription": transcription,
        "timestamp": time.time()
    }
    es.index(index=INDEX_NAME, document=doc)


# ------------------- Monitoring Thread -------------------
def monitor():
    while True:
        cpu = psutil.cpu_percent(percpu=True)
        gpu = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        print(f"CPU per core: {cpu} | GPU mem: {gpu:.1f} MB")
        time.sleep(2)


# ------------------- Main Ray Pipeline -------------------
def main():
    threading.Thread(target=monitor, daemon=True).start()

    # Clean any old Ray sessions and disable problematic subsystems
    os.environ["RAY_DISABLE_WINDOWS_VERSION_CHECK"] = "1"
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
    os.environ["RAY_USE_MULTIPROCESSING_START_METHOD"] = "spawn"
    os.environ["RAY_memory_MONITOR_REFRESH_MS"] = "0"

    # Make sure no stale Ray background process exists
    try:
        ray.shutdown()
    except Exception:
        pass

    start = time.time()

    # 🔧 Safe init: in-process only (no background daemons)
    ray.init( local_mode=True,# include_dashboard=True,
              ignore_reinit_error=True,
              num_cpus=8,
              num_gpus=1 )
    print(f"🚀 Ray initialized with {ray.cluster_resources()} resources")

    # ---------------- Stage 1: Download & extract ----------------
    df = pd.read_csv(CSV_PATH)
    df = df[(df["follower_count"] > MIN_FOLLOWERS) & (df["video_count"] > MIN_VIDEO_COUNT)]
    df = df.head(NUM_CREATORS)

    cpu_tasks = [process_creator.remote(row._asdict()) for row in df.itertuples(index=False)]
    completed = ray.get(cpu_tasks)
    print(f"✅ Stage 1 completed for {len(completed)} creators")

    # ---------------- Stage 2: Transcription ----------------
    whisper = WhisperActor.remote()
    creators = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ])
    transcription_tasks = [
        (c, whisper.transcribe_creator.remote(os.path.join(RESULTS_DIR, c)))
        for c in creators
    ]

    transcriptions = {}
    for c, fut in transcription_tasks:
        text = ray.get(fut)
        transcriptions[c] = text
        save_to_elasticsearch(c, text)  # ✅ Save each transcription to Elasticsearch

    ensure_dir(TRANSCRIPTIONS_DIR)
    pd.DataFrame(list(transcriptions.items()), columns=["creator", "transcription"]).to_csv(
        os.path.join(TRANSCRIPTIONS_DIR, "creator_transcriptions.csv"), index=False
    )
    print(f"✅ Stage 2 done, saved transcriptions")

    # ---------------- Stage 3: Embeddings + FAISS ----------------
    embedder = EmbeddingActor.remote()
    emb_results = []
    for c, text in transcriptions.items():
        emb_results.append(ray.get(embedder.embed_creator.remote(c, text, RESULTS_DIR)))

    creators_list, img_embs, txt_embs = zip(*emb_results)
    embedder.build_faiss_index.remote(np.stack(img_embs), np.stack(txt_embs), list(creators_list))

    print(f"\n⏱️ Total pipeline time: {time.time() - start:.2f}s")
    ray.shutdown()


if __name__ == "__main__":
    main()