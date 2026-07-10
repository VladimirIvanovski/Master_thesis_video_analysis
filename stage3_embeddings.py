"""
Stage 3 — GPU CLIP embeddings (in-memory JPEG, cross-creator batching)
====================================================================
All frames from a creator batch are encoded in shared GPU forward() passes.
"""

import os
import glob
import time
import ray
import torch
import numpy as np

from config import (RESULTS_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED,
                    IMAGE_WEIGHT, TEXT_WEIGHT, DEVICE, EMBEDDER_GPU_FRAC,
                    CLIP_GPU_BATCH_SIZE)


def _vram_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return round(torch.cuda.memory_allocated() / 1e6, 1)
    return 0.0


@ray.remote(num_gpus=EMBEDDER_GPU_FRAC, num_cpus=0)
class GPUBatchEmbeddingActor:
    def __init__(self):
        t0 = time.time()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval().to(DEVICE).half()
        self._clip_batch = CLIP_GPU_BATCH_SIZE
        self._init_s = round(time.time() - t0, 2)
        self._init_vram_mb = _vram_mb()

    def warmup(self) -> dict:
        return {"ready": True, "init_s": self._init_s, "vram_mb": self._init_vram_mb}

    def _jpeg_to_tensor(self, jpeg: bytes):
        try:
            from torchvision.io import decode_jpeg
            t = decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8))
            if t.shape[0] == 4:
                t = t[:3]
            t = t.float() / 255.0
            if t.shape[-1] != 224 or t.shape[-2] != 224:
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0), size=(224, 224), mode="bilinear", antialias=True
                ).squeeze(0)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            return (t - mean) / std
        except Exception:
            return None

    def embed_creators_batch(self, items: list[dict]) -> list:
        """
        Embed a batch of creators. All frames across all creators are encoded
        together in shared GPU forward() passes (not creator-by-creator).
        """
        t0 = time.time()
        n = len(items)
        if n == 0:
            return []

        tensor_owner: list[int] = []
        all_tensors: list = []

        for ci, item in enumerate(items):
            for jpeg in item.get("frames") or []:
                t = self._jpeg_to_tensor(jpeg)
                if t is not None:
                    all_tensors.append(t)
                    tensor_owner.append(ci)

        img_embs = np.zeros((n, 512), dtype="float32")
        if all_tensors:
            all_feats = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for i in range(0, len(all_tensors), self._clip_batch):
                    batch = torch.stack(all_tensors[i:i + self._clip_batch]).to(DEVICE).half()
                    all_feats.append(self.model.encode_image(batch).cpu())
            all_feats = torch.cat(all_feats).numpy()
            owner_arr = np.array(tensor_owner)
            for ci in range(n):
                mask = owner_arr == ci
                if not mask.any():
                    continue
                feats = all_feats[mask]
                norms = np.linalg.norm(feats, axis=1, keepdims=True)
                feats = feats / np.maximum(norms, 1e-8)
                img_embs[ci] = feats.mean(axis=0).astype("float32")

        texts = [it["text"] for it in items]
        txt_embs = np.zeros((n, 512), dtype="float32")
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, n, 32):
                batch_texts = texts[i:i + 32]
                tokens = self.tokenizer(batch_texts).to(DEVICE)
                txt = self.model.encode_text(tokens)
                txt = txt / txt.norm(dim=-1, keepdim=True)
                txt_embs[i:i + len(batch_texts)] = txt.cpu().numpy().astype("float32")

        return [
            (items[i]["creator"], img_embs[i], txt_embs[i])
            for i in range(n)
        ]

    def build_faiss_index(self, emb_results: list, transcriptions: dict) -> list:
        import faiss

        if not emb_results:
            return []

        creators, img_embs, txt_embs = zip(*emb_results)
        img_stack = np.stack(img_embs)
        txt_stack = np.stack(txt_embs)
        n = len(creators)

        combined = IMAGE_WEIGHT * img_stack + TEXT_WEIGHT * txt_stack
        norms = np.linalg.norm(combined, axis=1)
        img_norms = np.linalg.norm(img_stack, axis=1)
        txt_norms = np.linalg.norm(txt_stack, axis=1)
        valid_mask = (img_norms > 0.01) & (norms > 0.01) & (txt_norms > 0.01)

        def _meaningful(text):
            text = str(text).strip()
            if len(text) < 50:
                return False
            words = text.lower().split()
            if not words:
                return False
            top = max(set(words), key=words.count)
            return words.count(top) / len(words) <= 0.4

        meaningful = np.array([_meaningful(transcriptions.get(c, "")) for c in creators])
        valid_mask &= meaningful
        valid_idx = np.where(valid_mask)[0]

        combined_valid = combined[valid_idx]
        if len(combined_valid) > 0:
            combined_valid /= np.linalg.norm(combined_valid, axis=1, keepdims=True)
        creators_valid = [creators[i] for i in valid_idx]

        index = faiss.IndexFlatIP(512)
        if len(combined_valid) > 0:
            index.add(combined_valid)

        faiss.write_index(index, "creators.index")
        with open("creators.txt", "w") as f:
            f.write("\n".join(creators_valid))
        np.save("image_embs.npy", img_stack[valid_idx])
        np.save("text_embs.npy", txt_stack[valid_idx])

        return list(zip(creators, img_stack.tolist(), txt_stack.tolist()))

    def embed_all_creators_inmemory(
        self, transcriptions: dict, creator_frames: dict[str, list[bytes]],
    ) -> list:
        items = [
            {"creator": c, "text": transcriptions[c], "frames": creator_frames.get(c, [])}
            for c in sorted(transcriptions.keys())
        ]
        emb = self.embed_creators_batch(items)
        return self.build_faiss_index(emb, transcriptions)

    def embed_all_creators(self, transcriptions: dict, root_dir: str = RESULTS_DIR) -> list:
        creators = sorted(transcriptions.keys())
        items = []
        for creator in creators:
            frames = glob.glob(os.path.join(root_dir, creator, "**/frames/*.*"), recursive=True)
            frames = [f for f in frames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            jpegs = []
            for f in frames:
                try:
                    with open(f, "rb") as fh:
                        jpegs.append(fh.read())
                except Exception:
                    pass
            items.append({"creator": creator, "text": transcriptions[creator], "frames": jpegs})
        emb = self.embed_creators_batch(items)
        return self.build_faiss_index(emb, transcriptions)
