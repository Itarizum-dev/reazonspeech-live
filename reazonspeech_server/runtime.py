import asyncio
import logging

import numpy as np

from .vad import SileroSegmenter

logger = logging.getLogger(__name__)
HEAD_PADDING = np.zeros(int(16000 * 0.3), dtype=np.float32)
TAIL_PADDING = np.zeros(int(16000 * 0.5), dtype=np.float32)


def model_files(precision):
    encoder_suffix = ".int8.onnx" if precision != "fp32" else ".onnx"
    decoder_suffix = ".onnx" if precision == "int8-fp32" else encoder_suffix
    return {
        "tokens": "tokens.txt",
        "encoder": f"encoder-epoch-99-avg-1{encoder_suffix}",
        "decoder": f"decoder-epoch-99-avg-1{decoder_suffix}",
        "joiner": f"joiner-epoch-99-avg-1{encoder_suffix}",
    }


class ReazonRuntime:
    model_name = "reazonspeech-k2-v2"

    def __init__(self, settings, recognizer, sherpa_onnx):
        self.settings = settings
        self.recognizer = recognizer
        self.sherpa_onnx = sherpa_onnx
        self.inference_lock = asyncio.Lock()

    @classmethod
    def load(cls, settings):
        import sherpa_onnx
        from huggingface_hub import hf_hub_download

        logger.info("モデルロード開始: %s (%s)", cls.model_name, settings.model_precision)
        model_paths = {
            key: hf_hub_download(
                repo_id="reazon-research/reazonspeech-k2-v2",
                filename=filename,
                local_files_only=True,
            )
            for key, filename in model_files(settings.model_precision).items()
        }
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=model_paths["tokens"],
            encoder=model_paths["encoder"],
            decoder=model_paths["decoder"],
            joiner=model_paths["joiner"],
            num_threads=settings.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        logger.info("モデルロード完了: %s", cls.model_name)
        return cls(settings, recognizer, sherpa_onnx)

    def create_segmenter(self):
        return SileroSegmenter(self.settings, self.sherpa_onnx)

    async def transcribe(self, samples):
        async with self.inference_lock:
            return await asyncio.to_thread(self._transcribe, samples)

    def _transcribe(self, samples):
        # The transducer drops or garbles speech that starts or ends without surrounding silence.
        padded = np.concatenate((HEAD_PADDING, np.asarray(samples, dtype=np.float32), TAIL_PADDING))
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, padded)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()
