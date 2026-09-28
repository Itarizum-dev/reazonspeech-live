from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 9090
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    vad_max_speech_seconds: int = 30
    vad_model: str = "/models/silero_vad.onnx"
    model_precision: str = "int8"
    num_threads: int = 2

    @classmethod
    def from_env(cls):
        settings = cls(
            host=os.getenv("HOST", cls.host),
            port=int(os.getenv("PORT", cls.port)),
            vad_threshold=float(os.getenv("VAD_THRESHOLD", cls.vad_threshold)),
            vad_min_speech_ms=int(os.getenv("VAD_MIN_SPEECH_MS", cls.vad_min_speech_ms)),
            vad_min_silence_ms=int(os.getenv("VAD_MIN_SILENCE_MS", cls.vad_min_silence_ms)),
            vad_max_speech_seconds=int(os.getenv("VAD_MAX_SPEECH_SECONDS", cls.vad_max_speech_seconds)),
            vad_model=os.getenv("VAD_MODEL", cls.vad_model),
            model_precision=os.getenv("MODEL_PRECISION", cls.model_precision),
            num_threads=int(os.getenv("NUM_THREADS", cls.num_threads)),
        )
        settings.validate()
        return settings

    def validate(self):
        if not 0.0 < self.vad_threshold < 1.0:
            raise ValueError("VAD_THRESHOLD must be between 0 and 1")
        if self.vad_min_speech_ms < 0 or self.vad_min_silence_ms < 0:
            raise ValueError("VAD duration settings must be non-negative")
        if self.vad_max_speech_seconds <= 0:
            raise ValueError("VAD_MAX_SPEECH_SECONDS must be positive")
        if self.model_precision not in {"int8", "fp32", "int8-fp32"}:
            raise ValueError("MODEL_PRECISION must be int8, fp32, or int8-fp32")
