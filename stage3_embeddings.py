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

    def build_faiss_index(self, image_embs, text_embs, creators):
        """Combine embeddings and build FAISS index."""
        combined = (IMAGE_WEIGHT * image_embs + TEXT_WEIGHT * text_embs)
        combined /= np.linalg.norm(combined, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(512)
        index.add(combined)
        faiss.write_index(index, "creators.index")
        with open("creators.txt", "w") as f:
            f.write("\n".join(creators))
        np.save("image_embs.npy", image_embs)
        np.save("text_embs.npy", text_embs)
        print(f"✅ FAISS index built with {len(creators)} creators.")