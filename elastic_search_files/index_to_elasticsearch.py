import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from config import ES_PASSWORD, CSV_PATH, IMAGE_WEIGHT, TEXT_WEIGHT

password = ES_PASSWORD

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", password),
    verify_certs=False,
    ssl_show_warn=False
)

INDEX_NAME = "creator_transcriptions"

# Get parent directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
df = pd.read_csv(os.path.join(PARENT_DIR, "transcriptions/creator_transcriptions.csv"))
creators = [c.strip() for c in open(os.path.join(PARENT_DIR, "creators.txt")).read().splitlines()]
img_embs = np.load(os.path.join(PARENT_DIR, "image_embs.npy"))
txt_embs = np.load(os.path.join(PARENT_DIR, "text_embs.npy"))
profile_df = pd.read_csv(CSV_PATH)

# Create index mapping
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

es.indices.create(
    index=INDEX_NAME,
    mappings={
        "properties": {
            "username": {"type": "keyword"},
            "transcription": {"type": "text"},
            "image_embedding": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"},
            "text_embedding": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"},
            "combined_embedding": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"},
            "follower_count": {"type": "long"},
            "video_count": {"type": "long"},
            "indexed_at": {"type": "date"}
        }
    }
)

# Index data
import time
from datetime import datetime

transcription_dict = dict(zip(df["creator"], df["transcription"]))
profile_dict = profile_df.set_index("username").to_dict("index")
indexed_count = 0
skipped_count = 0

for i, username in enumerate(creators):
    transcription = transcription_dict.get(username, "")
    
    # Handle NaN values - convert to empty string
    if pd.isna(transcription):
        transcription = ""
    
    # Check if embeddings are zero vectors
    img_emb = img_embs[i]
    txt_emb = txt_embs[i]
    
    img_magnitude = np.linalg.norm(img_emb)
    txt_magnitude = np.linalg.norm(txt_emb)
    
    # Skip if either embedding is zero magnitude
    if img_magnitude == 0 or txt_magnitude == 0:
        print(f"⚠️  Skipping {username} - zero magnitude embedding")
        skipped_count += 1
        continue
    
    # Create combined embedding (like FAISS uses)
    combined_emb = IMAGE_WEIGHT * img_emb + TEXT_WEIGHT * txt_emb
    combined_emb /= np.linalg.norm(combined_emb)
    
    # Get profile metadata
    profile = profile_dict.get(username, {})
    
    doc = {
        "username": username,
        "transcription": transcription,
        "image_embedding": img_emb.tolist(),
        "text_embedding": txt_emb.tolist(),
        "combined_embedding": combined_emb.tolist(),
        "follower_count": profile.get("follower_count", 0),
        "video_count": profile.get("video_count", 0),
        "indexed_at": datetime.now().isoformat()
    }
    
    try:
        es.index(index=INDEX_NAME, document=doc, id=username)
        indexed_count += 1
        print(f"✓ Indexed {username}")
    except Exception as e:
        print(f"❌ Error indexing {username}: {e}")
        skipped_count += 1

print(f"\n✅ Successfully indexed {indexed_count} creators to Elasticsearch")
print(f"⚠️  Skipped {skipped_count} creators (zero embeddings or errors)")