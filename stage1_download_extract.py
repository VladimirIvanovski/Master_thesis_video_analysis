"""
Stage 1 — In-memory FFmpeg extract (zero intermediate disk I/O)
================================================================
Audio kept as raw s16le PCM bytes (half the size of float32 arrays).
Frames kept as JPEG bytes. Parallel audio+video FFmpeg pipes per task.
"""

import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray

from config import (
    RESULTS_DIR,
    MAX_VIDEOS_PER_CREATOR,
    AUDIO_MAX_SEC,
    FFMPEG_THREADS,
    get_random_proxy,
)
from utils import ensure_dir

SAMPLE_RATE = 16000
FRAME_COUNT = 10
FRAME_SIZE = 224


def _ffmpeg_audio_pcm(mp4_path: str, max_sec: int = AUDIO_MAX_SEC) -> bytes:
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-threads", str(FFMPEG_THREADS),
        "-ss", "1", "-i", mp4_path,
        "-t", str(max_sec),
        "-map", "0:a", "-vn",
        "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return proc.stdout if proc.returncode == 0 else b""


def _ffmpeg_frames_mjpeg(mp4_path: str, count: int = FRAME_COUNT) -> bytes:
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-threads", str(FFMPEG_THREADS),
        "-ss", "1", "-t", "10", "-i", mp4_path,
        "-map", "0:v",
        "-vf", f"fps=1,scale={FRAME_SIZE}:{FRAME_SIZE}",
        "-qscale:v", "3",
        "-frames:v", str(count),
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return proc.stdout if proc.returncode == 0 else b""


def parse_mjpeg_stream(data: bytes, max_frames: int = FRAME_COUNT) -> list[bytes]:
    frames: list[bytes] = []
    i = 0
    while i < len(data) and len(frames) < max_frames:
        start = data.find(b"\xff\xd8", i)
        if start == -1:
            break
        end = data.find(b"\xff\xd9", start + 2)
        if end == -1:
            break
        frames.append(data[start:end + 2])
        i = end + 2
    return frames


def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert s16le PCM to float32 — call only inside GPU Whisper worker."""
    if not pcm:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def extract_video_to_memory(mp4_path: str) -> dict:
    """
    Slim Ray payload — audio as raw PCM bytes (not float32), frames as JPEG bytes.
    """
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pcm_fut = pool.submit(_ffmpeg_audio_pcm, mp4_path)
        vid_fut = pool.submit(_ffmpeg_frames_mjpeg, mp4_path)
        pcm = pcm_fut.result()
        mjpeg = vid_fut.result()

    frames = parse_mjpeg_stream(mjpeg)
    return {
        "mp4": mp4_path,
        "creator": os.path.basename(os.path.dirname(mp4_path)),
        "audio_pcm": pcm,
        "frames": frames,
        "has_audio": len(pcm) > 0,
        "n_frames": len(frames),
        "s1_time": round(time.time() - t0, 3),
    }


@ray.remote(num_cpus=1)
def extract_video_to_memory_remote(mp4_path: str) -> dict:
    return extract_video_to_memory(mp4_path)


# ── Legacy disk helpers ───────────────────────────────────────────────────────

@ray.remote(num_cpus=1)
def download_video(url: str, output_path: str, proxy: str = None, max_retries: int = 4):
    for attempt in range(max_retries + 1):
        try:
            command = ["yt-dlp", "-q", "-o", output_path]
            if proxy:
                command += ["--proxy", proxy]
            command.append(url)
            subprocess.run(
                command, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return output_path
        except subprocess.CalledProcessError:
            if attempt < max_retries:
                print(f"Download failed, retrying ({attempt + 1}/{max_retries})...")
            else:
                print(f"Download failed after {max_retries + 1} attempts: {url}")
    return None


@ray.remote(num_cpus=1)
def extract_audio(video_path: str, audio_path: str):
    try:
        pcm = _ffmpeg_audio_pcm(video_path)
        if not pcm:
            return None
        with open(audio_path, "wb") as f:
            f.write(pcm)
        return audio_path
    except Exception:
        return None


@ray.remote(num_cpus=1)
def extract_10_frames_cpu(video_path, save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = parse_mjpeg_stream(_ffmpeg_frames_mjpeg(video_path))
        for idx, jpeg in enumerate(frames):
            with open(os.path.join(save_dir, f"{base_name}_frame_{idx:02d}.jpg"), "wb") as f:
                f.write(jpeg)
        return save_dir if frames else None
    except Exception as e:
        print(f"Frame extraction error {video_path}: {e}")
        return None


@ray.remote(num_cpus=1)
def download_and_extract(url: str, video_path: str, proxy: str = None) -> dict:
    for attempt in range(5):
        try:
            command = ["yt-dlp", "-q", "-o", video_path]
            if proxy:
                command += ["--proxy", proxy]
            command.append(url)
            subprocess.run(
                command, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            payload = extract_video_to_memory(video_path)
            payload["creator"] = os.path.basename(os.path.dirname(video_path))
            return payload
        except subprocess.CalledProcessError:
            if attempt >= 4:
                break
    return {
        "mp4": video_path,
        "creator": os.path.basename(os.path.dirname(video_path)),
        "audio_pcm": b"",
        "frames": [],
        "has_audio": False,
        "n_frames": 0,
        "s1_time": 0.0,
    }


@ray.remote(num_cpus=10)
def process_creator(row_dict):
    username = row_dict["username"]
    video_ids = eval(row_dict["video_ids"])
    durations = eval(row_dict["video_durations_list"])
    valid = [
        vid for vid, dur in zip(video_ids, durations)
        if 10 <= dur <= 50
    ][:MAX_VIDEOS_PER_CREATOR]
    creator_dir = ensure_dir(os.path.join(RESULTS_DIR, username))
    proxy = get_random_proxy()
    ray.get([
        download_and_extract.remote(
            f"https://www.tiktok.com/@{username}/video/{vid}",
            os.path.join(creator_dir, f"{vid}.mp4"),
            proxy,
        )
        for vid in valid
    ])
    return username
