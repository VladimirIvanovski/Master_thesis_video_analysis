from elasticsearch import Elasticsearch
from config import ES_PASSWORD

# Connect to Elasticsearch
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False
)

INDEX_NAME = "user_interactions"

# Create index for user interactions
if es.indices.exists(index=INDEX_NAME):
    print(f"Deleting existing index: {INDEX_NAME}")
    es.indices.delete(index=INDEX_NAME)

es.indices.create(
    index=INDEX_NAME,
    mappings={
        "properties": {
            "username": {"type": "keyword"},
            "liked_creator": {"type": "keyword"},
            "query_context": {"type": "text"},
            "label": {"type": "keyword"},
            "timestamp": {"type": "date"}
        }
    }
)

print(f"✅ Created Elasticsearch index: {INDEX_NAME}")
print("This index will store user likes for personalized recommendations")
