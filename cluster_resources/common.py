"""
Shared helpers for the Task 1 (best embedding combination) and Task 2
(clustering quality) evaluation scripts.

Loads the creator embeddings + CLIP text model that already exist at the
project root (produced by the main pipeline), and builds FAISS
IndexFlatIP indices for any visual/text weight combination.
"""
import os
import sys

import faiss
import numpy as np
import open_clip
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE, ES_PASSWORD

CREATORS_PATH = os.path.join(ROOT, "creators.txt")
IMAGE_EMBS_PATH = os.path.join(ROOT, "image_embs.npy")
TEXT_EMBS_PATH = os.path.join(ROOT, "text_embs.npy")

# The 5 embedding-combination configs evaluated in Task 1.
# (name, image_weight, text_weight)
CONFIGS = [
    ("visual_only", 1.00, 0.00),
    ("text_only", 0.00, 1.00),
    ("50_50", 0.50, 0.50),
    ("85_15", 0.85, 0.15),  # current production config
    ("15_85", 0.15, 0.85),
]

QUERIES = ["makeup", "fitness", "cooking", "fashion", "tech"]
TOP_K = 10

# Some queries were saved under slightly different text in the Flask demo
# (e.g. "makeup creators" instead of "makeup"). Map each test query to every
# ES query_context string that should count as feedback for it.
QUERY_ALIASES = {
    "makeup": ["makeup", "makeup creators"],
    "fitness": ["fitness"],
    "cooking": ["cooking", "cooking content"],
    "fashion": ["fashion"],
    "tech": ["tech"],
}


def normalize_query(q):
    return " ".join((q or "").strip().lower().split())


def load_es_feedback(username="vladimir"):
    """
    Reads the good/bad creator feedback already stored in Elasticsearch's
    `user_interactions` index (collected via the Flask demo apps) and returns
    {query: {"good": set(creators), "bad": set(creators)}} for each of the 5
    test QUERIES, merging in any aliased query_context variants.
    """
    from elasticsearch import Elasticsearch

    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", ES_PASSWORD),
        verify_certs=False,
        ssl_show_warn=False,
    )
    resp = es.search(
        index="user_interactions",
        query={"term": {"username": username}},
        size=1000,
        _source=["liked_creator", "query_context", "label"],
    )

    good_by_key = {}
    bad_by_key = {}
    for hit in resp["hits"]["hits"]:
        s = hit["_source"]
        creator = s.get("liked_creator")
        if not creator:
            continue
        key = normalize_query(s.get("query_context"))
        target = bad_by_key if (s.get("label") or "good").lower() == "bad" else good_by_key
        target.setdefault(key, set()).add(creator)

    feedback = {}
    for query in QUERIES:
        good, bad = set(), set()
        for alias in QUERY_ALIASES.get(query, [query]):
            key = normalize_query(alias)
            good |= good_by_key.get(key, set())
            bad |= bad_by_key.get(key, set())
        feedback[query] = {"good": good, "bad": bad}
    return feedback


def load_creators_and_embeddings():
    """Loads creator names + their raw visual/text CLIP embeddings."""
    creators = [c.strip() for c in open(CREATORS_PATH, encoding="utf-8").read().splitlines() if c.strip()]
    img_embs = np.load(IMAGE_EMBS_PATH).astype("float32")
    txt_embs = np.load(TEXT_EMBS_PATH).astype("float32")
    return creators, img_embs, txt_embs


def build_combined_embeddings(img_embs, txt_embs, image_weight, text_weight):
    """Weighted sum of visual + text embeddings, re-normalized to unit length."""
    combined = image_weight * img_embs + text_weight * txt_embs
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid div-by-zero for any all-zero rows
    return (combined / norms).astype("float32")


def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


_clip_model = None
_clip_tokenizer = None


def load_clip_text_model():
    """Lazily loads the CLIP model used to embed text queries (shared across configs)."""
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE)
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        _clip_model = model.eval().to(DEVICE).half()
    return _clip_model, _clip_tokenizer


def embed_query(query):
    """Encodes a text query into a normalized CLIP text embedding, shape (dim,)."""
    model, tokenizer = load_clip_text_model()
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokens = tokenizer([query]).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().astype("float32")
