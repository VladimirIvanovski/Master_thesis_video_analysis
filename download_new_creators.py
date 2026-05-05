"""
download_new_creators.py
========================
Downloads new TikTok creators (not already in results_4) until
total video count reaches ~1000.

- Skips ALL usernames already in results_4/
- Uses relaxed filter (video_count > 10, any follower count)
- Downloads 3 videos per creator
- Stops as soon as we have enough to reach 1000 total videos
"""

import os, ray, pandas as pd
from config import CSV_PATH, RESULTS_DIR, MAX_VIDEOS_PER_CREATOR
from stage1_download_extract import process_creator

TARGET_TOTAL_VIDEOS = 1000
VIDEOS_PER_CREATOR  = MAX_VIDEOS_PER_CREATOR   # 3


def count_existing_videos() -> int:
    total = 0
    for creator in os.listdir(RESULTS_DIR):
        cdir = os.path.join(RESULTS_DIR, creator)
        if not os.path.isdir(cdir):
            continue
        mp4s = [f for f in os.listdir(cdir) if f.endswith(".mp4")]
        total += len(mp4s)
    return total


def main():
    # ── How many more videos do we need? ──────────────────────────────────────
    existing_videos   = count_existing_videos()
    already_done      = set(os.listdir(RESULTS_DIR))
    videos_needed     = max(0, TARGET_TOTAL_VIDEOS - existing_videos)
    creators_needed   = -(-videos_needed // VIDEOS_PER_CREATOR)  # ceiling div

    print(f"Existing videos   : {existing_videos}")
    print(f"Target            : {TARGET_TOTAL_VIDEOS}")
    print(f"Videos needed     : {videos_needed}")
    print(f"Creators to fetch : {creators_needed}")

    if videos_needed <= 0:
        print("Already at or above target. Nothing to do.")
        return

    # ── Load CSV, skip already-downloaded, use relaxed filter ─────────────────
    df = pd.read_csv(CSV_PATH)
    df = df[df["video_count"] > 10]                    # relaxed — any size creator
    df = df[~df["username"].isin(already_done)]        # skip already downloaded
    df = df.head(creators_needed)                      # take only what we need

    print(f"Selected {len(df)} new creators from CSV\n")
    if df.empty:
        print("No new creators found in CSV.")
        return

    # ── Init Ray ──────────────────────────────────────────────────────────────
    ray.init(
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
        include_dashboard=False,
    )
    print(f"Ray ready | {ray.cluster_resources()}\n")

    # ── Download in parallel ───────────────────────────────────────────────────
    tasks = [
        process_creator.remote(row._asdict())
        for row in df.itertuples(index=False)
    ]

    completed = 0
    for fut in tasks:
        username = ray.get(fut)
        completed += 1
        print(f"  [{completed}/{len(tasks)}] done: {username}")

    ray.shutdown()

    # ── Final count ───────────────────────────────────────────────────────────
    new_total = count_existing_videos()
    print(f"\nDone. Total videos now: {new_total}")


if __name__ == "__main__":
    main()
