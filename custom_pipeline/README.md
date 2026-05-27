# Custom Pipeline

Run the full video pipeline (download → frames → audio → transcription) for specific TikTok video IDs.

## Usage

1. Open `run_custom.py`
2. Add your videos as `(username, video_id)` pairs to the `VIDEOS` list at the top:

```python
VIDEOS = [
    ("therock", "7123456789012345678"),
    ("charlidamelio", "7234567890123456789"),
]
```

3. Run from the project root:

```bash
python custom_pipeline/run_custom.py
```

## Output structure

```
custom_pipeline/
  videos/
    <creator>/
      <video_id>/
        <video_id>.mp4
        <video_id>.wav
        frames/
          <video_id>_frame_01.png
          ...
  transcriptions/
    <video_id>.txt     ← one file per video
```

## Notes

- The script auto-resolves the creator username from TikTok using yt-dlp.
- All steps are checkpointed: if a file already exists it won't re-download/re-extract.
- The main pipeline (`results_4/`, `transcriptions/`) is **not touched**.
- Requires Ray, faster-whisper, ffmpeg, and yt-dlp to be installed.
