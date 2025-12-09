import os, subprocess, ray
from config import RESULTS_DIR, MAX_VIDEOS_PER_CREATOR
from utils import ensure_dir

# ======================================================
# CPU Stage: Download, extract audio & frames
# ======================================================

@ray.remote(num_cpus=1)
def download_video(url: str, output_path: str):
    """Download a single video using yt-dlp."""
    try:
        subprocess.run(
            ["yt-dlp", "-q", "-o", output_path, url],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return output_path
    except subprocess.CalledProcessError:
        return None


@ray.remote(num_cpus=1)
def extract_audio(video_path: str, audio_path: str):
    """Extract audio (mono 16kHz WAV) using ffmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    except Exception:
        return None


@ray.remote(num_cpus=1)
def extract_10_frames_cpu(video_path, save_dir):
    """
    Extract up to 10 frames (1 fps from seconds 1–10) from a given video file.
    Works safely on Windows by ensuring the folder exists and escaping paths properly.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Use absolute paths to avoid Windows relative path issues
        abs_video = os.path.abspath(video_path)
        abs_out = os.path.abspath(os.path.join(save_dir, f"{base_name}_frame_%02d.png"))

        # Run ffmpeg synchronously
        command = [
            "ffmpeg",
            "-y",              # overwrite output files
            "-ss", "1",        # start at 1 second
            "-t", "10",        # capture up to 10 seconds
            "-i", abs_video,
            "-vf", "fps=1",    # 1 frame per second
            "-qscale:v", "2",  # quality scale
            abs_out
        ]

        # Ensure the process completes and prints errors if any
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"❌ Frame extraction failed for {video_path}\n{result.stderr[:200]}")
            return None

        # Count extracted frames
        frames = [f for f in os.listdir(save_dir) if f.endswith(".png")]
        print(f"🖼️ Extracted {len(frames)} frames from {base_name}")
        return save_dir

    except Exception as e:
        print(f"❌ Frame extraction error {video_path}: {e}")
        return None


@ray.remote(num_cpus=2)
def process_creator(row_dict):
    """Full CPU pipeline per creator: download, extract audio/frames."""
    username = row_dict["username"]
    video_ids = eval(row_dict["video_ids"])
    durations = eval(row_dict["video_durations_list"])

    valid = [vid for vid, dur in zip(video_ids, durations) if 10 <= dur <= 50][:MAX_VIDEOS_PER_CREATOR]
    creator_dir = ensure_dir(os.path.join(RESULTS_DIR, username))

    tasks = []
    for vid in valid:
        url = f"https://www.tiktok.com/@{username}/video/{vid}"
        video_path = os.path.join(creator_dir, f"{vid}.mp4")
        audio_path = os.path.join(creator_dir, f"{vid}.wav")
        frames_dir = os.path.join(creator_dir, vid, "frames")

        v = download_video.remote(url, video_path)
        tasks.append(extract_audio.remote(v, audio_path))
        tasks.append(extract_10_frames_cpu.remote(v, frames_dir))

    ray.get(tasks)
    return username