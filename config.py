# ======================================================
# Configuration constants (shared across all stages)
# ======================================================

import random

PROXIES = [
    "http://123.45.67.89:8080",
    "http://98.76.54.32:8000",
    "http://11.22.33.44:3128"
]

def get_random_proxy():
    return random.choice(PROXIES)

RESULTS_DIR = "results_4"
TRANSCRIPTIONS_DIR = "transcriptions"

# TikTok filtering
NUM_CREATORS = 10
MAX_VIDEOS_PER_CREATOR = 3
MIN_FOLLOWERS = 200_000
MIN_VIDEO_COUNT = 100

# Models
WHISPER_MODEL_SIZE = "tiny"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Compute
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MAX_CPU_WORKERS = 10

# Embedding weights
IMAGE_WEIGHT = 0.6
TEXT_WEIGHT = 0.4

# Input file
CSV_PATH = "C:/Users/vladimir/Downloads/tiktok_profile_5k.csv"

# Ray configuration
RAY_DASHBOARD = False