# ======================================================
# Configuration constants (shared across all stages)
# ======================================================

import os
import random

PROXIES = [p for p in os.environ.get("TIKTOK_PROXIES", "").split(",") if p]

def get_random_proxy():
    return random.choice(PROXIES) if PROXIES else None

RESULTS_DIR = "results_4"
TRANSCRIPTIONS_DIR = "transcriptions"
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")

# TikTok filtering
NUM_CREATORS = 350
MAX_VIDEOS_PER_CREATOR = 3
MIN_FOLLOWERS = 150_000
MIN_VIDEO_COUNT = 80

# Models
WHISPER_MODEL_SIZE = "small"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Compute
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MAX_CPU_WORKERS = 10

# Embedding weights
IMAGE_WEIGHT = 0.85
TEXT_WEIGHT = 0.15

# Input file
CSV_PATH = os.environ.get("CSV_PATH", "cluster_resources/thesis_357_creators.csv")

# Ray configuration
RAY_DASHBOARD = True

# ======================================================
# Pipeline & Concurrency Overrides
# ======================================================
AUDIO_MAX_SEC = 180      # Used by Stage 1 to cap audio extraction length
FFMPEG_THREADS = 1       # Keeps CPU usage per Stage 1 worker low
WHISPER_ACTORS = 1       # Single full-GPU actor (Stage 3 off → Whisper owns GPU)
WHISPER_BATCH_SIZE = 32  # Videos per Whisper Ray task
WHISPER_GPU_BATCH_SIZE = 16  # Audio chunks per GPU forward() pass
WHISPER_GPU_FRAC = 0.9   # GPU fraction for the Whisper actor
EMBEDDER_GPU_FRAC = 0.9  # GPU fraction for CLIP (runs alone after Whisper is killed)
S2_MAX_INFLIGHT = 2      # Concurrent Whisper batch tasks in flight
CLIP_GPU_BATCH_SIZE = 64 # JPEG frames per CLIP encode_image() call
S3_CREATOR_BATCH = 8     # Creators per CLIP Ray task (lower = starts sooner)
S3_MAX_INFLIGHT = 2      # Concurrent CLIP batch tasks
GPU_RELEASE_TIMEOUT = 90 # Seconds to wait for GPU after killing Whisper
RAY_ADDRESS = None       # Set to "auto" or cluster address for multi-node