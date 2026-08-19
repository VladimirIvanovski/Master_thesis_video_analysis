# Master thesis: scalable multimodal TikTok creator search
# (Whisper + CLIP + FAISS + Elasticsearch + Ray)

## What this repo is

Code and evaluation files for the master's thesis pipeline: download/extract → GPU Whisper → CLIP embeddings → FAISS / Elasticsearch → Flask search.

## Run the pipeline

```text
python pipeline_50_videos.py
```

Uses `cluster_resources/thesis_357_creators.csv` by default (`CSV_PATH`). MP4s go in `results_4/`. Set `ES_PASSWORD` if you index to Elasticsearch.

```text
python rebuild_faiss_index.py
python elastic_search_files/index_to_elasticsearch.py
python flask_apps/demo_app.py
```

Demo: http://localhost:5001

## Evaluation tables

See [`cluster_resources/README.md`](cluster_resources/README.md).
