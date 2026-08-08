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
            # beam_size=1 + no repetition guard was the fastest greedy-decoding
            # config, but it made the model highly prone to getting stuck in
            # repetition loops on music/near-silent audio (e.g. the same phrase
            # repeated 40+ times). BatchedInferencePipeline does not support
            # Whisper's temperature-fallback retry, but it does respect these
            # three knobs, so they are the cheapest way to curb repetition loops
            # while staying on the batched (fast) code path.
            beam_size=5,
            best_of=1,
            patience=1,
            length_penalty=1,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
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
            # without_timestamps=True forces the model to decode a full dense
            # 30s chunk as one continuous block of text with no internal
            # checkpoints, which causes severe truncation/failure on
            # information-dense speech (measured: a 30s chunk full of clear
            # speech decoded to just "Oh, yeah" with timestamps off, vs. the
            # full correct transcript with timestamps on). Enabling timestamps
            # lets the model self-segment into natural sub-segments, which
            # forward()/_split_segments_by_timestamps already knows how to
            # stitch back into one chunk_text per chunk.
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=False,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            # CRITICAL FIX: _ensure_tokenizer() below detects the spoken language
            # only ONCE, from the very first audio chunk the actor ever sees, then
            # caches and reuses that single language's tokenizer for every
            # subsequent video from every subsequent creator for the actor's
            # entire lifetime (actors are long-lived by design). With
            # multilingual=True, generate_segment_batched() instead detects the
            # language independently for every item in the batch and swaps in
            # the correct language token per item, so mixed-language batches are
            # transcribed correctly instead of being forced through one
            # arbitrary, globally "stuck" language.
            multilingual=True,
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

    def _windowed_chunks(self, audio: np.ndarray) -> list[tuple[np.ndarray, dict]]:
        """
        Split raw audio into fixed-size, non-overlapping windows of
        `chunk_length` seconds each (30s), one Whisper input per window.

        NOTE: this used to call faster_whisper.vad.collect_chunks(), but that
        function is meant to MERGE a list of pre-detected VAD speech segments
        into <=30s batches — not to split one giant "whole clip" range. Given
        a single oversized range it emits a bogus, fully-empty first "chunk"
        (which Whisper hallucinates on, e.g. "Rwy'n meddwl am ychydig") and
        then pad_or_trim() silently discards all audio beyond the first 30s
        of the real chunk. Manual fixed windowing avoids both problems.
        """
        from faster_whisper.audio import pad_or_trim

        if audio is None or audio.size == 0:
            return []
        sampling_rate = self.model.feature_extractor.sampling_rate
        window = int(self.model.feature_extractor.chunk_length * sampling_rate)
        out = []
        for start in range(0, audio.shape[0], window):
            chunk = audio[start:start + window]
            meta = {
                "offset": start / sampling_rate,
                "duration": chunk.shape[0] / sampling_rate,
                "segments": [],
            }
            feature = self.model.feature_extractor(chunk)[..., :-1]
            out.append((pad_or_trim(feature), meta))
        return out

    def _pcm_to_chunks(self, audio_pcm: bytes) -> list[tuple[np.ndarray, dict]]:
        if not audio_pcm:
            return []
        audio = pcm_bytes_to_float32(audio_pcm)
        del audio_pcm
        return self._windowed_chunks(audio)

    def _audio_to_chunks(self, audio: np.ndarray) -> list[tuple[np.ndarray, dict]]:
        return self._windowed_chunks(audio)

    @staticmethod
    def _keep_segment(seg: dict, options) -> bool:
        """
        BatchedInferencePipeline.forward() computes no_speech_prob/avg_logprob/
        compression_ratio per segment but never checks them against
        options.{no_speech,log_prob,compression_ratio}_threshold — that
        filtering only happens in the higher-level WhisperModel.transcribe(),
        which this pipeline doesn't use. Apply the same checks manually so
        near-silent/music-only chunks don't inject hallucinated text.
        """
        if options.no_speech_threshold is not None:
            should_skip = seg["no_speech_prob"] > options.no_speech_threshold
            if (
                options.log_prob_threshold is not None
                and seg["avg_logprob"] > options.log_prob_threshold
            ):
                should_skip = False
            if should_skip:
                return False
        if (
            options.compression_ratio_threshold is not None
            and seg["compression_ratio"] > options.compression_ratio_threshold
        ):
            return False
        return True

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
                        seg["text"].strip() for seg in result
                        if seg.get("text") and self._keep_segment(seg, options)
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
                        s["text"].strip() for s in result
                        if s.get("text") and self._keep_segment(s, options)
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
