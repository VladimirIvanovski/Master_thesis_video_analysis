import numpy as np
import faiss
import pandas as pd
from config import IMAGE_WEIGHT, TEXT_WEIGHT

# Load existing data
creators = [c.strip() for c in open("creators.txt").read().splitlines()]
img_embs = np.load("image_embs.npy")
txt_embs = np.load("text_embs.npy")
df = pd.read_csv("transcriptions/creator_transcriptions.csv")
transcriptions = dict(zip(df["creator"], df["transcription"]))

print(f"Loaded {len(creators)} creators")

# Combine embeddings
combined = (IMAGE_WEIGHT * img_embs + TEXT_WEIGHT * txt_embs)
norms = np.linalg.norm(combined, axis=1)
img_norms = np.linalg.norm(img_embs, axis=1)
txt_norms = np.linalg.norm(txt_embs, axis=1)

# Filter out creators with zero image embeddings, zero text embeddings, or very small combined embeddings
valid_mask = (img_norms > 0.01) & (norms > 0.01) & (txt_norms > 0.01)

# Also filter out creators with very short or repetitive transcriptions (likely generic/noise)
transcription_lengths = np.array([len(str(transcriptions.get(c, "")).strip()) for c in creators])

# Filter by length AND check for repetitive text (like "back back back" or "Thank you" repeated)
def is_meaningful_transcription(text):
    """Check if transcription is meaningful (not too short or repetitive)."""
    text = str(text).strip()
    if len(text) < 50:  # Increased minimum to 50 characters
        return False
    # Check for excessive repetition (same word repeated many times)
    words = text.lower().split()
    if len(words) > 0:
        most_common_word = max(set(words), key=words.count)
        repeat_ratio = words.count(most_common_word) / len(words)
        if repeat_ratio > 0.4:  # If one word is >40% of content, it's too repetitive
            return False
    return True

meaningful_mask = np.array([is_meaningful_transcription(transcriptions.get(c, "")) for c in creators])
valid_mask = valid_mask & (transcription_lengths >= 50) & meaningful_mask

valid_indices = np.where(valid_mask)[0]

if len(valid_indices) < len(creators):
    skipped = [creators[i] for i in np.where(~valid_mask)[0]]
    print(f"⚠️  Skipping {len(skipped)} creators with zero/small embeddings or short transcriptions: {skipped[:10]}")

combined_valid = combined[valid_indices]
combined_valid /= np.linalg.norm(combined_valid, axis=1, keepdims=True)
creators_valid = [creators[i] for i in valid_indices]

# Build FAISS index
index = faiss.IndexFlatIP(512)
index.add(combined_valid)
faiss.write_index(index, "creators.index")

# Save filtered data
with open("creators.txt", "w") as f:
    f.write("\n".join(creators_valid))
np.save("image_embs.npy", img_embs[valid_indices])
np.save("text_embs.npy", txt_embs[valid_indices])

print(f"✅ FAISS index rebuilt with {len(creators_valid)} creators (filtered {len(creators) - len(creators_valid)})")
