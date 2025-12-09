import os, glob, ray
from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, COMPUTE_TYPE, DEVICE

@ray.remote(num_gpus=1)
class WhisperActor:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print(f"🎧 Whisper model loaded ({WHISPER_MODEL_SIZE})")

    def transcribe_creator(self, creator_path: str) -> str:
        wavs = glob.glob(os.path.join(creator_path, "**/*.wav"), recursive=True)
        texts = []
        for w in wavs:
            try:
                segments, _ = self.model.transcribe(
                    w,
                    beam_size=1,
                    vad_filter=False,  # ← disables the problematic ONNX VAD
                    without_timestamps=True,
                )
                texts.append(" ".join(s.text.strip() for s in segments))

            except Exception as e:
                print(f"⚠️ Transcription failed for {w}: {e}")
        print("Transcription complete for",creator_path)
        return " ".join(texts)