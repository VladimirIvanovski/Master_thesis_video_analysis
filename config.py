# ======================================================
# Configuration constants (shared across all stages)
# ======================================================

RESULTS_DIR = "results_4"
TRANSCRIPTIONS_DIR = "transcriptions"

# TikTok filtering
NUM_CREATORS = 5
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
MAX_CPU_WORKERS = 6

# Embedding weights
IMAGE_WEIGHT = 0.6
TEXT_WEIGHT = 0.4

# Input file
CSV_PATH = "C:/Users/vladimir/Downloads/tiktok_profile_5k.csv"

# Ray configuration
RAY_DASHBOARD = False