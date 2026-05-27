# Custom Macedonian TikTok Transcription Pipeline

A standalone extension of the main TikTok creator analysis pipeline for targeted processing of **Macedonian-language TikTok videos**.

Given a list of TikTok URLs, this pipeline will:
1. **Download** the video (`yt-dlp`)
2. **Extract frames** — 1 frame/sec from seconds 1–10 (FFmpeg)
3. **Extract audio** — 16kHz mono WAV (FFmpeg)
4. **Transcribe** — `openai-whisper large`, language locked to Macedonian (`mk`)

All output is isolated in `custom_pipeline/` — the main pipeline is never touched.

---

## Quick Start

### 1. Install dependencies

```bash
pip install ray openai-whisper yt-dlp
```

FFmpeg must be installed and available on `PATH`.

### 2. Add your TikTok URLs

Open `macedonian_transcription_pipeline.py` and add full TikTok URLs to the `URLS` list:

```python
URLS = [
    "https://www.tiktok.com/@dinevv/video/7596645699547106571",
    "https://www.tiktok.com/@someuser/video/7234567890123456789",
]
```

### 3. Run

```bash
cd C:\Users\vladimir\PyCharmMiscProject
python custom_pipeline/macedonian_transcription_pipeline.py
```

---

## Output Structure

```
custom_pipeline/
  videos/
    <creator>/
      <video_id>/
        <video_id>.mp4       ← downloaded video
        <video_id>.wav       ← 16kHz mono audio
        frames/
          <video_id>_frame_01.png
          ...                ← up to 10 frames
  transcriptions/
    <video_id>.txt           ← Whisper transcription (Macedonian)
```

---

## Configuration

All settings are at the top of `run_custom.py`:

| Parameter | Default | Description |
|---|---|---|
| `URLS` | — | List of full TikTok URLs to process |
| `WHISPER_MODEL_SIZE` | `"large"` | Model size: `tiny` / `small` / `medium` / `large` / `large-v3` |
| `DEVICE` | `"cuda"` | Use `"cpu"` if no GPU available |

All settings are at the top of `macedonian_transcription_pipeline.py`.

To use the strongest available model:
```python
WHISPER_MODEL_SIZE = "large-v3"
```

---

## Why `large` instead of `tiny`/`small`?

For Macedonian — a lower-resource language — smaller models produce frequent word substitutions and dropped phrases on conversational audio:

| Model | Multilingual WER | Reverb penalty |
|---|---|---|
| tiny | ~12% | +15.5 pp |
| small | ~7% | +7.4 pp |
| medium | ~5% | +5.9 pp |
| **large** ← used | **~4%** | **+2.3 pp** |

Sources: [OpenWhispr benchmarks](https://openwhispr.com/blog/whisper-model-sizes-explained) · [Whisper-RIR-Mega, arXiv 2026](https://arxiv.org/abs/2603.02252)

---

## Checkpointing

Each step checks if output already exists before running:
- Video already downloaded → skip download
- WAV already extracted → skip audio extraction
- Frames already exist → skip frame extraction

To re-run a specific step, delete its output file and re-run `macedonian_transcription_pipeline.py`.

---

## Repository

[VladimirIvanovski/Master_thesis_video_analysis — macedonian-tiktok-transcription-pipeline](https://github.com/VladimirIvanovski/Master_thesis_video_analysis/tree/macedonian-tiktok-transcription-pipeline)
