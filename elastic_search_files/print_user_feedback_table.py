"""
Print a client's saved queries and Good/Bad creators from Elasticsearch (user_interactions).

Usage:
  python elastic_search_files/print_user_feedback_table.py
  python elastic_search_files/print_user_feedback_table.py vladimir
  python elastic_search_files/print_user_feedback_table.py vladimir --csv
"""
import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from elasticsearch import Elasticsearch
from config import ES_PASSWORD

INDEX = "user_interactions"


def normalize_query(q):
    return " ".join((q or "").strip().lower().split())


def main():
    ap = argparse.ArgumentParser(description="Print user feedback as a table (queries x good/bad).")
    ap.add_argument("username", nargs="?", default="vladimir", help="Client username (default: vladimir)")
    ap.add_argument("--csv", action="store_true", help="Print CSV for spreadsheet / Google Doc")
    args = ap.parse_args()
    user = args.username

    es = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", ES_PASSWORD),
        verify_certs=False,
        ssl_show_warn=False,
    )

    try:
        exists = es.indices.exists(index=INDEX)
    except Exception as e:
        print("Cannot connect to Elasticsearch at http://localhost:9200")
        print("Start Docker:  docker start es-local")
        print("Error: %s" % e)
        return

    if not exists:
        print("Index '%s' does not exist. Start Elasticsearch and use the Flask demo first." % INDEX)
        return

    r = es.search(
        index=INDEX,
        size=1000,
        query={"term": {"username": user}},
        sort=[{"timestamp": {"order": "asc"}}],
        _source=["liked_creator", "query_context", "label", "timestamp"],
    )

    hits = r["hits"]["hits"]
    if not hits:
        print("No feedback documents for username=%r in %s." % (user, INDEX))
        return

    # query_key -> {"good": set(), "bad": set(), "queries_raw": set()}
    by_query = defaultdict(lambda: {"good": set(), "bad": set(), "raw": set()})

    for h in hits:
        s = h["_source"]
        creator = s.get("liked_creator")
        raw_q = s.get("query_context") or ""
        key = normalize_query(raw_q)
        if not creator or not key:
            continue
        by_query[key]["raw"].add(raw_q.strip())
        label = (s.get("label") or "good").lower()
        if label == "bad":
            by_query[key]["bad"].add(creator)
        else:
            by_query[key]["good"].add(creator)

    rows = []
    for key in sorted(by_query.keys()):
        g = by_query[key]
        raw_display = next(iter(sorted(g["raw"])), key)
        good_list = ", ".join(sorted(g["good"]))
        bad_list = ", ".join(sorted(g["bad"]))
        rows.append((raw_display, good_list, bad_list, len(g["good"]), len(g["bad"])))

    if args.csv:
        print("query,good_creators,bad_creators,good_count,bad_count")
        for raw_display, good_list, bad_list, ng, nb in rows:
            def esc(x):
                return '"' + str(x).replace('"', '""') + '"'
            print(",".join([esc(raw_display), esc(good_list), esc(bad_list), str(ng), str(nb)]))
        print("\n# Total queries with feedback: %d | Total docs scanned: %d" % (len(rows), len(hits)))
        return

    # --- Pretty table ---
    print("=" * 100)
    print("CLIENT: %s  |  INDEX: %s  |  DOCUMENTS: %d" % (user, INDEX, len(hits)))
    print("=" * 100)
    print()
    print("Saved queries (normalized for matching) and creators marked Good / Bad in the Flask demo.")
    print()

    col_q = 28
    col_good = 32
    col_bad = 32

    def trunc(s, w):
        s = s or ""
        return (s[: w - 2] + "..") if len(s) > w else s

    header = (
        "| " + "Query".ljust(col_q)
        + " | " + "Good creators".ljust(col_good)
        + " | " + "Bad creators".ljust(col_bad)
        + " |"
    )
    sep = "|" + "-" * (col_q + 2) + "|" + "-" * (col_good + 2) + "|" + "-" * (col_bad + 2) + "|"
    print(header)
    print(sep)
    for raw_display, good_list, bad_list, _, _ in rows:
        # wrap long cells: simple one-line trunc for terminal; full data in CSV
        line = (
            "| " + trunc(raw_display, col_q).ljust(col_q)
            + " | " + trunc(good_list, col_good).ljust(col_good)
            + " | " + trunc(bad_list, col_bad).ljust(col_bad)
            + " |"
        )
        print(line)

    print()
    print("Tip: run with --csv for full lists without truncation (paste into Google Sheets).")
    print("=" * 100)


if __name__ == "__main__":
    main()
