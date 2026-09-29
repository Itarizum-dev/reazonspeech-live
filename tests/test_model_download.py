import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reazonspeech_server.config import Settings
from reazonspeech_server.runtime import model_files


class ModelDownloadTests(unittest.TestCase):
    def test_precision_selects_exactly_the_required_asr_files(self):
        expected = {
            "int8": (".int8.onnx", ".int8.onnx", ".int8.onnx"),
            "fp32": (".onnx", ".onnx", ".onnx"),
            "int8-fp32": (".int8.onnx", ".onnx", ".int8.onnx"),
        }
        for precision, suffixes in expected.items():
            with self.subTest(precision=precision):
                files = model_files(precision)
                self.assertEqual(files["tokens"], "tokens.txt")
                self.assertEqual(
                    tuple(files[key] for key in ("encoder", "decoder", "joiner")),
                    tuple(f"{key}-epoch-99-avg-1{suffix}" for key, suffix in zip(("encoder", "decoder", "joiner"), suffixes)),
                )

    def test_download_reuses_cached_models_and_vad(self):
        from reazonspeech_server.download_models import download_models

        with tempfile.TemporaryDirectory() as directory:
            vad = Path(directory, "silero_vad.onnx")
            vad.write_bytes(b"cached")
            hub_download = Mock()
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hub_download)}), patch(
                "reazonspeech_server.download_models.urlopen"
            ) as urlopen:
                download_models(Settings(vad_model=str(vad), model_precision="int8-fp32"))

        self.assertEqual(hub_download.call_count, 4)
        self.assertTrue(all(call.kwargs["local_files_only"] for call in hub_download.call_args_list))
        self.assertCountEqual(
            [call.kwargs["filename"] for call in hub_download.call_args_list],
            list(model_files("int8-fp32").values()),
        )
        urlopen.assert_not_called()

    def test_missing_vad_is_downloaded_atomically(self):
        from reazonspeech_server.download_models import download_models

        with tempfile.TemporaryDirectory() as directory:
            vad = Path(directory, "silero_vad.onnx")
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=Mock())}), patch(
                "reazonspeech_server.download_models.urlopen"
            ) as urlopen:
                urlopen.return_value.__enter__.return_value.read.side_effect = [b"model", b""]
                download_models(Settings(vad_model=str(vad)))
            self.assertEqual(vad.read_bytes(), b"model")
            self.assertEqual(list(Path(directory).iterdir()), [vad])

    def test_only_missing_asr_file_uses_network(self):
        from reazonspeech_server.download_models import download_models

        def fetch(repo_id, filename, **kwargs):
            if filename.startswith("decoder") and kwargs.get("local_files_only"):
                raise FileNotFoundError(filename)
            return f"/models/{filename}"

        hub_download = Mock(side_effect=fetch)
        with tempfile.TemporaryDirectory() as directory:
            vad = Path(directory, "silero_vad.onnx")
            vad.write_bytes(b"cached")
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hub_download)}):
                download_models(Settings(vad_model=str(vad)))

        network_calls = [call for call in hub_download.call_args_list if not call.kwargs.get("local_files_only")]
        self.assertEqual(len(network_calls), 1)
        self.assertEqual(network_calls[0].kwargs["filename"], "decoder-epoch-99-avg-1.int8.onnx")

    def test_empty_vad_download_does_not_replace_model(self):
        from reazonspeech_server.download_models import download_models

        with tempfile.TemporaryDirectory() as directory:
            vad = Path(directory, "silero_vad.onnx")
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=Mock())}), patch(
                "reazonspeech_server.download_models.urlopen"
            ) as urlopen:
                urlopen.return_value.__enter__.return_value.read.return_value = b""
                with self.assertRaisesRegex(ValueError, "empty"):
                    download_models(Settings(vad_model=str(vad)))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
