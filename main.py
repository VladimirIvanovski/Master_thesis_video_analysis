import faiss
import torch
import numpy as np
import open_clip
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE


# ============================================================
# 1️⃣ LOAD FAISS INDEX + CREATOR METADATA
# ============================================================
INDEX_PATH = "creators.index"
CREATORS_PATH = "creators.txt"

index = faiss.read_index(INDEX_PATH)
creators = [c.strip() for c in open(CREATORS_PATH).read().splitlines()]
print(f"✅ Loaded FAISS index with {len(creators)} creators.")


# ============================================================
# 2️⃣ LOAD CLIP MODEL
# ============================================================
print(f"🧠 Loading CLIP model: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
model.eval().to(DEVICE).half()
print("✅ CLIP model ready for text search.")


# ============================================================
# 3️⃣ TEXT SEARCH FUNCTION
# ============================================================
def search_by_text(query: str, top_k: int = 5):
    """Find top_k creators semantically similar to a text query."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokens = tokenizer([query]).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features[0].cpu().numpy().astype("float32")

    scores, idxs = index.search(np.expand_dims(query_vec, axis=0), top_k)

    print("\n🔎 TEXT QUERY:", query)
    print("=" * 60)
    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
        print(f"{rank:>2}. {creators[idx]} — score={score:.4f}")
    print("=" * 60 + "\n")


# ============================================================
# 4️⃣ MAIN
# ============================================================
if __name__ == "__main__":
    # Example searches
    # search_by_text("White Dress", top_k=5)
    # search_by_text("fitness influencer lifting weights", top_k=5)
    # search_by_text("travel vlogger in Europe", top_k=5)

    from elasticsearch import Elasticsearch
    #
    es = Elasticsearch("http://localhost:9200")
    index_name = "creator_transcriptions"

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"🗑️ Index '{index_name}' deleted successfully.")
    else:
        print(f"ℹ️ Index '{index_name}' does not exist.")