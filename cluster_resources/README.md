# Supplementary evaluation files

These files support the experiments in Chapter 4 of the thesis.

## Thesis subset (357 creators)

[`thesis_357_creators.csv`](thesis_357_creators.csv) is the experimental corpus: every creator in `transcriptions/pipeline_streaming_transcriptions.csv`, matched to the Influencers Club profile export. All original columns are kept (`video_ids`, `video_durations_list`, `follower_count`, `video_desc_list`, `video_count`, `username`), plus `tiktok_url`.

The vendor export of ~5,000 profiles is **not** published here (research permission only). This file is the 357-creator subset used for download, transcription, and indexing.

## Other evaluation tables

- [`task1_labeling.csv`](task1_labeling.csv) — Precision@10 labels (config, query, rank, creator, score, relevant).
- [`task2_creator_clusters.csv`](task2_creator_clusters.csv) — cluster id per indexed creator (k = 10, visual-only).
