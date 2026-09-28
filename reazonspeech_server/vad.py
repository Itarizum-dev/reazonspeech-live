import numpy as np


class SileroSegmenter:
    """Per-connection Silero VAD state backed by sherpa-onnx."""

    def __init__(self, settings, sherpa_onnx):
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = settings.vad_model
        config.silero_vad.threshold = settings.vad_threshold
        config.silero_vad.min_speech_duration = settings.vad_min_speech_ms / 1000
        config.silero_vad.min_silence_duration = settings.vad_min_silence_ms / 1000
        config.silero_vad.max_speech_duration = settings.vad_max_speech_seconds
        config.sample_rate = 16000
        config.num_threads = 1
        if not config.validate():
            raise ValueError("Invalid Silero VAD configuration")
        self.window_size = config.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=settings.vad_max_speech_seconds + 1
        )
        self.pending = np.empty(0, dtype=np.float32)

    def accept(self, samples):
        self.pending = np.concatenate((self.pending, samples))
        completed = []
        while len(self.pending) >= self.window_size:
            window = self.pending[: self.window_size]
            self.pending = self.pending[self.window_size :]
            self.vad.accept_waveform(window)
            while not self.vad.empty():
                segment = self.vad.front
                audio = np.asarray(segment.samples, dtype=np.float32).copy()
                start = segment.start / 16000
                completed.append((audio, start, start + len(audio) / 16000))
                self.vad.pop()
        return completed
