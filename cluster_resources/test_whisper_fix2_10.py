"""
Ad-hoc check: re-runs the ACTUAL production Stage 1 + Stage 2 (now with the
windowing fix [no more misuse of collect_chunks / no more silent 30s+ content
loss] and without_timestamps=False [fixes severe truncation on dense 30s
chunks]) on a sample of creators, comparing against the transcriptions
currently in transcriptions/pipeline_streaming_transcriptions.csv (i.e. the
prior beam_size=5/repetition_penalty fix, before this round of fixes).

Run:
    python test_whisper_fix2_10.py
"""
import os
import sys

import pandas as pd
import ray

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from stage1_download_extract import extract_video_to_memory
from stage2_transcribe import GPUWhisperActor, aggregate_creator_transcriptions

RESULTS_DIR = os.path.join(ROOT, "results_4")
OLD_CSV = os.path.join(ROOT, "transcriptions", "pipeline_streaming_transcriptions.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_fix2_check.txt")

# rolopolotv = the specific case reported by the user (30s of clear English
# lost to "Oh, yeah"); the rest were previously affected by the
# "Rwy'n meddwl am ychydig" hallucination or are known-good controls.
CREATORS = [
    "rolopolotv", "1amnotjamaila", "ridukushmidahit", "filhadorei539",
    "drareejzulfiqar", "carrieberkk", "da.rafiki", "alabialasela4",
    "9baii7", "amanda_hadzimichalis", "adam.digital",
]


def main():
    ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)

    creator_videos = {}
    mp4_paths = []
    for creator in CREATORS:
        cdir = os.path.join(RESULTS_DIR, creator)
        vids = sorted(f for f in os.listdir(cdir) if f.endswith(".mp4"))
        creator_videos[creator] = [os.path.join(cdir, v) for v in vids]
        mp4_paths.extend(creator_videos[creator])

    print(f"Extracting audio+frames for {len(mp4_paths)} videos across {len(CREATORS)} creators...")
    extracted = [extract_video_to_memory(p) for p in mp4_paths]

    actor = GPUWhisperActor.remote()
    ray.get(actor.warmup.remote())
    result = ray.get(actor.transcribe_batch_memory.remote(extracted))
    ray.get(actor.shutdown.remote())

    video_texts = result["texts"]
    new_per_creator = aggregate_creator_transcriptions(video_texts, creator_videos, CREATORS)

    old_df = pd.read_csv(OLD_CSV).set_index("creator")["transcription"].to_dict()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for creator in CREATORS:
            f.write(f"\n{'=' * 80}\n{creator}\n{'=' * 80}\n")
            f.write("--- OLD (before windowing + timestamps fix) ---\n")
            f.write(str(old_df.get(creator, "<not found>"))[:800] + "\n")
            f.write("\n--- NEW (fixed windowing + without_timestamps=False) ---\n")
            f.write(str(new_per_creator.get(creator, "<empty>"))[:800] + "\n")

    print(f"Wrote comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()
