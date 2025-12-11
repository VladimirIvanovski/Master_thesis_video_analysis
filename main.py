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
    # search_by_text("indian person", top_k=5)
    # search_by_text("fitness influencer lifting weights", top_k=5)
    # search_by_text("travel vlogger in Europe", top_k=5)

    import os
    from minio import Minio

    # Connect to MinIO
    client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    BUCKET = "frames"


    def upload_all_frames(root_folder):
        """
        Scan all subfolders, find /frames/, and upload images to S3 bucket
        preserving creator → video → frame structure.
        """
        for creator in os.listdir(root_folder):
            creator_path = os.path.join(root_folder, creator)
            if not os.path.isdir(creator_path):
                continue

            # inside each creator folder
            for video_id in os.listdir(creator_path):
                video_path = os.path.join(creator_path, video_id)
                if not os.path.isdir(video_path):
                    continue

                frames_path = os.path.join(video_path, "frames")
                if not os.path.isdir(frames_path):
                    continue

                # upload each frame
                for img_file in os.listdir(frames_path):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        local_file = os.path.join(frames_path, img_file)

                        # S3 path
                        object_name = f"{creator}/{video_id}/{img_file}"

                        client.fput_object(BUCKET, object_name, local_file)
                        print(f"Uploaded → {object_name}")


    # Run it
    upload_all_frames("results_4")

