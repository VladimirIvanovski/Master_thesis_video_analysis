import os, glob, torch, numpy as np, faiss, ray
import open_clip
from PIL import Image
from config import (RESULTS_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED,
                    IMAGE_WEIGHT, TEXT_WEIGHT, DEVICE)

@ray.remote(num_gpus=1)
class EmbeddingActor:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval().to(DEVICE).half()
        print(f"🧠 CLIP model loaded ({CLIP_MODEL_NAME})")

    def embed_creator(self, creator_name, transcription, root_dir=RESULTS_DIR):
        """Generate mean image and text embeddings for one creator."""
        frames = glob.glob(os.path.join(root_dir, creator_name, "**/frames/*.*"), recursive=True)
        frames = [f for f in frames if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # ---- Image embeddings ----
        img_emb = np.zeros(512, dtype="float32")
        if frames:
            tensors = []
            for f in frames:
                try:
                    img = Image.open(f).convert("RGB")
                    tensors.append(self.preprocess(img))
                except Exception:
                    continue
            if tensors:
                feats = []
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for i in range(0, len(tensors), 64):
                        batch = torch.stack(tensors[i:i+64]).to(DEVICE).half()
                        e = self.model.encode_image(batch)
                        feats.append(e)
                feats = torch.cat(feats)
                feats /= feats.norm(dim=-1, keepdim=True)
                img_emb = feats.mean(dim=0).cpu().numpy().astype("float32")

        # ---- Text embedding ----
        txt_emb = np.zeros(512, dtype="float32")
        if transcription.strip():
            with torch.no_grad(), torch.cuda.amp.autocast():
                tokens = self.tokenizer([transcription]).to(DEVICE)
                txt = self.model.encode_text(tokens)
                txt /= txt.norm(dim=-1, keepdim=True)
                txt_emb = txt[0].cpu().numpy().astype("float32")
        print("embeddings completed for creator ",creator_name)
        return creator_name, img_emb, txt_emb

    def build_faiss_index(self, image_embs, text_embs, creators, transcriptions=None):
        """Combine embeddings and build FAISS index."""
        combined = (IMAGE_WEIGHT * image_embs + TEXT_WEIGHT * text_embs)
        norms = np.linalg.norm(combined, axis=1)
        img_norms = np.linalg.norm(image_embs, axis=1)
        txt_norms = np.linalg.norm(text_embs, axis=1)
        
        # Filter out creators with zero image embeddings or very small combined embeddings
        valid_mask = (img_norms > 0.01) & (norms > 0.01) & (txt_norms > 0.01)
        
        # Also filter out creators with very short or repetitive transcriptions
        if transcriptions:
            def is_meaningful(text):
                text = str(text).strip()
                if len(text) < 50:
                    return False
                # Check for excessive repetition
                words = text.lower().split()
                if len(words) > 0:
                    most_common_word = max(set(words), key=words.count)
                    repeat_ratio = words.count(most_common_word) / len(words)
                    if repeat_ratio > 0.4:  # >40% repetition is too much
                        return False
                return True
            
            meaningful_mask = np.array([is_meaningful(transcriptions.get(c, "")) for c in creators])
            valid_mask = valid_mask & meaningful_mask
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < len(creators):
            skipped = [creators[i] for i in np.where(~valid_mask)[0]]
            print(f"⚠️  Skipping {len(skipped)} creators with zero/small embeddings or short transcriptions: {skipped[:10]}")
        
        combined_valid = combined[valid_indices]
        combined_valid /= np.linalg.norm(combined_valid, axis=1, keepdims=True)
        creators_valid = [creators[i] for i in valid_indices]
        
        index = faiss.IndexFlatIP(512)
        index.add(combined_valid)
        faiss.write_index(index, "creators.index")
        with open("creators.txt", "w") as f:
            f.write("\n".join(creators_valid))
        np.save("image_embs.npy", image_embs[valid_indices])
        np.save("text_embs.npy", text_embs[valid_indices])
        print(f"✅ FAISS index built with {len(creators_valid)} creators (filtered {len(creators) - len(creators_valid)}).")