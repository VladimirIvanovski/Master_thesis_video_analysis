"""
Ad-hoc check: re-runs the ACTUAL production Stage 1 (FFmpeg extraction) +
Stage 2 (GPUWhisperActor, now with the repetition-loop fix: beam_size=5,
repetition_penalty=1.3, no_repeat_ngram_size=3) on 10 creators' existing raw
mp4s in results_4/, so the new transcriptions can be compared against the old
ones already in transcriptions/pipeline_streaming_transcriptions.csv.

Includes 6 creators previously found to have hallucinated/looping
transcriptions, and 2 known-good controls, so we can see whether the fix
helps the bad ones without breaking the good ones.

Run:
    python test_whisper_fix_10.py
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
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_fix_check.txt")

CREATORS = [
    "1amnotjamaila", "ridukushmidahit", "filhadorei539", "drareejzulfiqar",
    "carrieberkk", "da.rafiki", "alabialasela4", "9baii7",
    "amanda_hadzimichalis", "adam.digital",  # known-good controls
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
            f.write(f"--- OLD (beam_size=1, no repetition guard) ---\n")
            f.write(str(old_df.get(creator, "<not found>"))[:600] + "\n")
            f.write(f"\n--- NEW (beam_size=5, repetition_penalty=1.3, no_repeat_ngram_size=3) ---\n")
            f.write(str(new_per_creator.get(creator, "<empty>"))[:600] + "\n")

    print(f"Wrote comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()
