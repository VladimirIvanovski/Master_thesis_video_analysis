"""
Stage 2 — GPU Whisper (in-memory PCM, BatchedInferencePipeline, no temp files)
=============================================================================
Audio arrives as s16le PCM bytes; float32 conversion happens once inside the
GPU worker. Never writes temporary WAV files.
"""

import os
import time
import ray
import torch
import numpy as np

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_GPU_FRAC,
    WHISPER_GPU_BATCH_SIZE,
)
from stage1_download_extract import pcm_bytes_to_float32


def aggregate_creator_transcriptions(
    video_texts: dict[str, str],
    creator_videos: dict[str, list[str]],
    creators: list[str],
) -> dict[str, str]:
    creator_parts: dict[str, dict[str, str]] = {c: {} for c in creators}
    for mp4, text in video_texts.items():
        creator = os.path.basename(os.path.dirname(mp4))
        if creator in creator_parts:
            creator_parts[creator][mp4] = text
    return {
        c: " ".join(creator_parts[c].get(p, "") for p in sorted(creator_videos[c]))
        for c in creators
    }


@ray.remote(num_gpus=WHISPER_GPU_FRAC, num_cpus=0)
class GPUWhisperActor:
    def __init__(self, gpu_batch_size: int = WHISPER_GPU_BATCH_SIZE):
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = "float16" if device == "cuda" else "int8"
        self.device = device
        self.gpu_batch_size = gpu_batch_size

        model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=ctype)
        self.model = model
        self.pipe = BatchedInferencePipeline(model)
        self._language: str | None = None
        self._tokenizer = None

    def warmup(self) -> bool:
        return True

    def shutdown(self) -> dict:
        """Release Whisper model and CUDA memory before actor kill."""
        import gc
        t0 = time.time()
        vram_before = 0.0
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / 1e6
            try:
                del self.model
                del self.pipe
            except Exception:
                pass
            self.model = None
            self.pipe = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e6
        else:
            vram_after = 0.0
        return {
            "released": True,
            "vram_before_mb": round(vram_before, 1),
            "vram_after_mb": round(vram_after, 1),
            "shutdown_s": round(time.time() - t0, 2),
        }

    def _build_options(self, tokenizer):
        from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens

        return TranscriptionOptions(
            beam_size=1,
            best_of=1,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
            without_timestamps=True,
            max_initial_timestamp=0.0,
            word_timestamps=False,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps=[],
            hallucination_silence_threshold=None,
            hotwords=None,
        )

    def _ensure_tokenizer(self, feature: np.ndarray):
        if self._tokenizer is not None:
            return
        from faster_whisper.tokenizer import Tokenizer

        language = self._language
        if language is None:
            if not self.model.model.is_multilingual:
                language = "en"
            else:
                language, _, _ = self.model.detect_language(
                    features=feature,
                    language_detection_segments=1,
                    language_detection_threshold=0.5,
                )
            self._language = language

        self._tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=self._language,
        )

    def _pcm_to_chunks(self, audio_pcm: bytes) -> list[tuple[np.ndarray, dict]]:
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.vad import collect_chunks

        if not audio_pcm:
            return []

        audio = pcm_bytes_to_float32(audio_pcm)
        del audio_pcm

        sampling_rate = self.model.feature_extractor.sampling_rate
        chunk_length = self.model.feature_extractor.chunk_length
        clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
        audio_chunks, chunks_metadata = collect_chunks(
            audio, clip_timestamps, max_duration=chunk_length
        )
        del audio
        if not audio_chunks:
            return []

        out = []
        for chunk, meta in zip(audio_chunks, chunks_metadata):
            feature = self.model.feature_extractor(chunk)[..., :-1]
            out.append((pad_or_trim(feature), meta))
        return out

    def _audio_to_chunks(self, audio: np.ndarray) -> list[tuple[np.ndarray, dict]]:
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.vad import collect_chunks

        if audio is None or audio.size == 0:
            return []
        chunk_length = self.model.feature_extractor.chunk_length
        clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
        audio_chunks, chunks_metadata = collect_chunks(
            audio, clip_timestamps, max_duration=chunk_length
        )
        if not audio_chunks:
            return []
        return [
            (pad_or_trim(self.model.feature_extractor(c)[..., :-1]), m)
            for c, m in zip(audio_chunks, chunks_metadata)
        ]

    def transcribe_batch_memory(self, items: list[dict]) -> dict:
        """
        Transcribe slim payloads: {mp4, audio_pcm: bytes, frames}.
        Returns texts keyed by mp4; frames passed through for CLIP (audio dropped).
        """
        t0 = time.time()
        work_items: list[tuple[str, int, np.ndarray, dict]] = []
        failed: set[str] = set()
        frame_out: dict[str, list[bytes]] = {}

        for item in items:
            mp4 = item["mp4"]
            frame_out[mp4] = item.get("frames") or []
            try:
                chunks = self._pcm_to_chunks(item.get("audio_pcm") or b"")
                if not chunks:
                    failed.add(mp4)
                    continue
                for chunk_idx, (feature, meta) in enumerate(chunks):
                    work_items.append((mp4, chunk_idx, feature, meta))
            except Exception:
                failed.add(mp4)

        mp4_list = [item["mp4"] for item in items]
        texts: dict[str, list[tuple[int, str]]] = {p: [] for p in mp4_list}

        if work_items:
            self._ensure_tokenizer(work_items[0][2])
            options = self._build_options(self._tokenizer)
            features = [w[2] for w in work_items]
            metas = [w[3] for w in work_items]

            for i in range(0, len(features), self.gpu_batch_size):
                batch_feat = np.stack(features[i:i + self.gpu_batch_size])
                batch_meta = metas[i:i + self.gpu_batch_size]
                batch_items = work_items[i:i + self.gpu_batch_size]
                results = self.pipe.forward(
                    batch_feat, self._tokenizer, batch_meta, options
                )
                for witem, result in zip(batch_items, results):
                    mp4, chunk_idx, _, _ = witem
                    chunk_text = " ".join(
                        seg["text"].strip() for seg in result if seg.get("text")
                    )
                    texts[mp4].append((chunk_idx, chunk_text))

        out: dict[str, str] = {}
        for mp4 in mp4_list:
            if mp4 in failed or not texts.get(mp4):
                out[mp4] = ""
            else:
                ordered = sorted(texts[mp4], key=lambda x: x[0])
                out[mp4] = " ".join(t for _, t in ordered if t)

        return {
            "texts": out,
            "frames": frame_out,
            "s2_time": round(time.time() - t0, 3),
            "n_files": len(items),
        }

    def transcribe_batch(self, wav_paths: list[str]) -> dict:
        """Legacy disk path."""
        from faster_whisper.audio import decode_audio

        t0 = time.time()
        work_items = []
        failed = set()
        for wav_path in wav_paths:
            try:
                audio = decode_audio(
                    wav_path, sampling_rate=self.model.feature_extractor.sampling_rate
                )
                chunks = self._audio_to_chunks(audio)
                if not chunks:
                    failed.add(wav_path)
                    continue
                for idx, (feat, meta) in enumerate(chunks):
                    work_items.append((wav_path, idx, feat, meta))
            except Exception:
                failed.add(wav_path)

        texts = {p: [] for p in wav_paths}
        if work_items:
            self._ensure_tokenizer(work_items[0][2])
            options = self._build_options(self._tokenizer)
            for i in range(0, len(work_items), self.gpu_batch_size):
                batch = work_items[i:i + self.gpu_batch_size]
                feats = np.stack([b[2] for b in batch])
                metas = [b[3] for b in batch]
                results = self.pipe.forward(feats, self._tokenizer, metas, options)
                for witem, result in zip(batch, results):
                    key, idx, _, _ = witem
                    texts[key].append((idx, " ".join(
                        s["text"].strip() for s in result if s.get("text")
                    )))

        out = {}
        for key in wav_paths:
            if key in failed or not texts[key]:
                out[key] = ""
            else:
                out[key] = " ".join(t for _, t in sorted(texts[key]))

        return {"texts": out, "s2_time": round(time.time() - t0, 3), "n_files": len(wav_paths)}

    def transcribe(self, wav_path: str) -> dict:
        result = self.transcribe_batch([wav_path])
        return {"text": result["texts"].get(wav_path, ""), "s2_time": result["s2_time"]}


WhisperActor = GPUWhisperActor
