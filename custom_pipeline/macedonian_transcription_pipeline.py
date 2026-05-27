"""
Custom Pipeline — Run full video pipeline for specific TikTok URLs
==================================================================
Paste full TikTok URLs into the URLS list below, then run:

    python custom_pipeline/run_custom.py

For each URL this script will:
  1. Download the video
  2. Extract audio (16kHz mono WAV) and frames (1fps, first 10s)
  3. Transcribe audio with openai-whisper (strong model, auto language detection)
  4. Save transcription to custom_pipeline/transcriptions/<video_id>.txt

All output lives in custom_pipeline/ — nothing in the main pipeline is touched.
"""

import os
import sys
import re
import subprocess

import ray
import whisper

# ---------------------------------------------------------------------------
# ADD YOUR TIKTOK URLs HERE
# ---------------------------------------------------------------------------
URLS = [
    "https://www.tiktok.com/@dinevv/video/7596645699547106571",
    "https://www.tiktok.com/@dinevv/video/7534752469625310469",
    # Add more full URLs here...
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR         = os.path.join(BASE_DIR, "videos")
TRANSCRIPTIONS_DIR = os.path.join(BASE_DIR, "transcriptions")

WHISPER_MODEL_SIZE = "medium"   # options: tiny, base, small, medium, large
DEVICE             = "cuda"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_url(url: str) -> tuple[str, str]:
    """
    Extract (creator, video_id) from a full TikTok URL.
    e.g. https://www.tiktok.com/@dinevv/video/7642348063700290836?
         -> ("dinevv", "7642348063700290836")
    """
    match = re.search(r"tiktok\.com/@([^/]+)/video/(\d+)", url)
    if not match:
        raise ValueError(f"Cannot parse TikTok URL: {url}")
    return match.group(1), match.group(2)


# ---------------------------------------------------------------------------
# Ray tasks
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
def download_video(url: str, output_path: str, max_retries: int = 4):
    """Download a single TikTok video using yt-dlp."""
    for attempt in range(max_retries + 1):
        try:
            subprocess.run(
                ["yt-dlp", "-q", "-o", output_path, url],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return output_path
        except subprocess.CalledProcessError:
            if attempt < max_retries:
                print(f"  [WARN] Download failed, retrying ({attempt+1}/{max_retries})...")
            else:
                print(f"  [ERROR] Download failed after {max_retries+1} attempts: {url}")
    return None


@ray.remote(num_cpus=1)
def extract_audio(video_path: str, audio_path: str):
    """Extract 16kHz mono WAV from video."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    except Exception as e:
        print(f"  [ERROR] Audio extraction failed for {video_path}: {e}")
        return None


@ray.remote(num_cpus=1)
def extract_frames(video_path: str, frames_dir: str):
    """Extract 1 frame/sec from seconds 1-10 into frames_dir."""
    try:
        os.makedirs(frames_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        out  = os.path.abspath(os.path.join(frames_dir, f"{base}_frame_%02d.png"))
        result = subprocess.run([
            "ffmpeg", "-y", "-ss", "1", "-t", "10",
            "-i", os.path.abspath(video_path),
            "-vf", "fps=1", "-qscale:v", "2", out
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] Frame extraction failed: {result.stderr[:100]}")
            return None
        n = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        print(f"  Extracted {n} frames for {base}")
        return frames_dir
    except Exception as e:
        print(f"  [ERROR] Frame extraction error: {e}")
        return None


@ray.remote(num_gpus=1)
class WhisperActor:
    def __init__(self):
        self.model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
        print(f"  openai-whisper model loaded ({WHISPER_MODEL_SIZE})")

    def transcribe(self, audio_path: str) -> str:
        try:
            result = self.model.transcribe(audio_path, fp16=True, language="mk")
            return result["text"].strip()
        except Exception as e:
            print(f"  [WARN] Transcription failed for {audio_path}: {e}")
            return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not URLS:
        print("No URLs in URLS list. Add some TikTok URLs and re-run.")
        sys.exit(0)

    ensure_dir(VIDEOS_DIR)
    ensure_dir(TRANSCRIPTIONS_DIR)

    print("=" * 60)
    print(f"  CUSTOM PIPELINE — {len(URLS)} video(s)")
    print("=" * 60)

    # Parse URLs into metadata
    print("\n[1/4] Parsing URLs...")
    video_meta = []  # list of (url, video_id, creator, video_path, audio_path, frames_dir)
    for url in URLS:
        try:
            creator, vid = parse_url(url)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  @{creator} / {vid}")
        video_dir  = ensure_dir(os.path.join(VIDEOS_DIR, creator, vid))
        video_path = os.path.join(video_dir, f"{vid}.mp4")
        audio_path = os.path.join(video_dir, f"{vid}.wav")
        frames_dir = os.path.join(video_dir, "frames")
        video_meta.append((url, vid, creator, video_path, audio_path, frames_dir))

    # Step 2: Download + extract (Ray)
    print("\n[2/4] Downloading videos + extracting audio & frames (Ray)...")
    ray.init(ignore_reinit_error=True)

    download_futures = []
    for url, vid, creator, video_path, audio_path, frames_dir in video_meta:
        if os.path.exists(video_path):
            print(f"  {vid}: video already exists, skipping download.")
            download_futures.append(None)
            continue
        download_futures.append(download_video.remote(url, video_path))

    # Wait for downloads, then kick off audio/frame extraction
    extract_futures = []
    for i, (url, vid, creator, video_path, audio_path, frames_dir) in enumerate(video_meta):
        if download_futures[i] is not None:
            resolved_path = ray.get(download_futures[i])
            if resolved_path is None:
                print(f"  {vid}: download failed, skipping.")
                extract_futures.append((vid, None, None))
                continue

        af = extract_audio.remote(video_path, audio_path) if not os.path.exists(audio_path) else None
        ff = extract_frames.remote(video_path, frames_dir) if not os.path.exists(frames_dir) or not os.listdir(frames_dir) else None
        extract_futures.append((vid, af, ff))

    for vid, af, ff in extract_futures:
        if af: ray.get(af)
        if ff: ray.get(ff)
    print("  Done.")

    # Step 3: Transcribe
    print("\n[3/4] Transcribing audio (openai-whisper)...")
    whisper_actor = WhisperActor.remote()

    transcription_futures = []
    for url, vid, creator, video_path, audio_path, frames_dir in video_meta:
        if os.path.exists(audio_path):
            transcription_futures.append((vid, whisper_actor.transcribe.remote(audio_path)))
        else:
            print(f"  {vid}: no audio file found, skipping transcription.")
            transcription_futures.append((vid, None))

    # Step 4: Save transcriptions
    print("\n[4/4] Saving transcriptions...")
    for vid, future in transcription_futures:
        if future is None:
            continue
        text = ray.get(future)
        out_path = os.path.join(TRANSCRIPTIONS_DIR, f"{vid}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        preview = text[:80].replace("\n", " ") if text else "(empty)"
        print(f"  {vid} -> {out_path}")
        print(f"    Preview: {preview}...")

    print("\n" + "=" * 60)
    print("  Custom pipeline complete.")
    print(f"  Transcriptions saved to: {TRANSCRIPTIONS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
