import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elasticsearch import Elasticsearch
from config import ES_PASSWORD

password = ES_PASSWORD

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", password),
    verify_certs=False,
    ssl_show_warn=False
)

INDEX_NAME = "creator_transcriptions"

# Get all documents
results = es.search(index=INDEX_NAME, size=100)

print(f"\n📊 Total documents found: {results['hits']['total']['value']}\n")

for hit in results["hits"]["hits"]:
    print(hit)
    username = hit['_source']['username']
    transcription = hit['_source']['transcription'][:100] if hit['_source']['transcription'] else "No transcription"
    print(f"Username: {username}")
    print(f"Transcription: {transcription}...")
    print("---")