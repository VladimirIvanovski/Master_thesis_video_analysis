"""Print Elasticsearch index field names and types (columns). Run: python elastic_search_files/print_es_columns.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elasticsearch import Elasticsearch
from config import ES_PASSWORD

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)

INDICES = ["creator_transcriptions", "user_interactions"]


def describe_field(name, spec, indent=0):
    pad = "  " * indent
    typ = spec.get("type")
    if typ:
        bits = [typ]
        if "dims" in spec:
            bits.append("dims=%s" % spec["dims"])
        if "similarity" in spec:
            bits.append("similarity=%s" % spec["similarity"])
        print("%s%s: %s" % (pad, name, ", ".join(bits)))
        return
    nested = spec.get("properties")
    if nested:
        print("%s%s: object" % (pad, name))
        for k, v in sorted(nested.items()):
            describe_field(k, v, indent + 1)


def main():
    for idx in INDICES:
        print("=" * 60)
        print("INDEX: %s" % idx)
        print("=" * 60)
        if not es.indices.exists(index=idx):
            print("  (index does not exist — start Docker ES and index data first)\n")
            continue
        m = es.indices.get_mapping(index=idx)
        root = m[idx]["mappings"].get("properties", {})
        print("Fields (columns):")
        for field in sorted(root.keys()):
            describe_field(field, root[field], 1)
        # sample doc count
        try:
            c = es.count(index=idx)["count"]
            print("\nDocument count: %s" % c)
        except Exception as e:
            print("\nCount error: %s" % e)
        print()


if __name__ == "__main__":
    main()
