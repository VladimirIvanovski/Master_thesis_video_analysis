"""Print one creator_transcriptions document with all fields (embeddings truncated for screenshot)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from elasticsearch import Elasticsearch
from config import ES_PASSWORD

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)

INDEX = "creator_transcriptions"
PREVIEW = 8  # show first N floats of each vector


def shorten_vec(v):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)):
        return v
    n = len(v)
    head = v[:PREVIEW]
    return {"length": n, "first_%s_values" % PREVIEW: head, "note": "... truncated for display ..."}


def main():
    if not es.indices.exists(index=INDEX):
        print("Index %s does not exist." % INDEX)
        return
    r = es.search(index=INDEX, size=1, query={"match_all": {}})
    hits = r["hits"]["hits"]
    if not hits:
        print("No documents in %s." % INDEX)
        return
    src = hits[0]["_source"].copy()
    uname = src.get("username", "?")
    for key in ("image_embedding", "text_embedding", "combined_embedding"):
        if key in src and isinstance(src[key], list):
            src[key] = shorten_vec(src[key])
    print("=" * 70)
    print("ONE SAMPLE DOCUMENT — index: %s" % INDEX)
    print("username: %s" % uname)
    print("=" * 70)
    print(json.dumps(src, indent=2, ensure_ascii=False))
    print("=" * 70)
    print("(Embedding arrays shortened to first %s floats; full dim is 512.)" % PREVIEW)


if __name__ == "__main__":
    main()
