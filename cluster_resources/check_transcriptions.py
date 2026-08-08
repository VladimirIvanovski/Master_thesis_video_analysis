"""
Ad-hoc check: re-transcribes a handful of raw source videos directly with
faster-whisper (same settings as the pipeline: small model, GPU, fp16) and
prints per-segment no_speech_prob / avg_logprob, to verify whether the
existing pipeline transcriptions for these creators are genuine speech or
Whisper hallucinations on music/near-silent audio.
"""
import os

from faster_whisper import WhisperModel

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_4")

CREATORS = ["1amnotjamaila", "ridukushmidahit", "filhadorei539", "drareejzulfiqar", "carrieberkk"]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_output.txt")


def main():
    model = WhisperModel("small", device="cuda", compute_type="float16")
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        def log(msg):
            out.write(msg + "\n")
            out.flush()

        for creator in CREATORS:
            cdir = os.path.join(RESULTS_DIR, creator)
            for fname in sorted(f for f in os.listdir(cdir) if f.endswith(".mp4")):
                path = os.path.join(cdir, fname)
                log(f"\n=== {creator} / {fname} ===")
                try:
                    segments, info = model.transcribe(path, beam_size=5, vad_filter=False)
                    log(f"detected language: {info.language} (p={info.language_probability:.2f})")
                    for seg in segments:
                        log(f"  [{seg.start:5.1f}-{seg.end:5.1f}] no_speech_prob={seg.no_speech_prob:.2f} "
                            f"avg_logprob={seg.avg_logprob:.2f}  text={seg.text!r}")
                except Exception as e:
                    log(f"ERROR: {e!r}")


if __name__ == "__main__":
    main()
