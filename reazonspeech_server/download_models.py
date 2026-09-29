import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from .config import Settings
from .runtime import model_files


VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"


def download_models(settings):
    from huggingface_hub import hf_hub_download

    for filename in model_files(settings.model_precision).values():
        try:
            hf_hub_download(
                repo_id="reazon-research/reazonspeech-k2-v2",
                filename=filename,
                local_files_only=True,
            )
        except FileNotFoundError:
            hf_hub_download(
                repo_id="reazon-research/reazonspeech-k2-v2",
                filename=filename,
            )

    vad_path = Path(settings.vad_model)
    if vad_path.is_file() and vad_path.stat().st_size:
        return

    vad_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=vad_path.parent, delete=False) as output:
        temp_path = Path(output.name)
        try:
            with urlopen(VAD_URL, timeout=60) as response:
                shutil.copyfileobj(response, output)
            output.flush()
            if not temp_path.stat().st_size:
                raise ValueError("Downloaded Silero VAD model is empty")
            os.replace(temp_path, vad_path)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_models(Settings.from_env())
