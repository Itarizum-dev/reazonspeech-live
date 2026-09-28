import sys
import types
import unittest
from unittest.mock import Mock, patch

from reazonspeech_server.config import Settings
from reazonspeech_server.runtime import ReazonRuntime


class ModelLoadingTests(unittest.TestCase):
    def test_loads_only_the_selected_int8_model_files_once(self):
        download = Mock(side_effect=lambda repo_id, filename: f"/cache/{filename}")
        offline_recognizer = Mock()
        from_transducer = Mock(return_value=offline_recognizer)
        sherpa = types.SimpleNamespace(
            OfflineRecognizer=types.SimpleNamespace(from_transducer=from_transducer)
        )
        hub = types.SimpleNamespace(hf_hub_download=download)

        with patch.dict(sys.modules, {"sherpa_onnx": sherpa, "huggingface_hub": hub}):
            runtime = ReazonRuntime.load(Settings())

        self.assertIs(runtime.recognizer, offline_recognizer)
        self.assertEqual(download.call_count, 4)
        self.assertCountEqual(
            [call.kwargs["filename"] for call in download.call_args_list],
            [
                "tokens.txt",
                "encoder-epoch-99-avg-1.int8.onnx",
                "decoder-epoch-99-avg-1.int8.onnx",
                "joiner-epoch-99-avg-1.int8.onnx",
            ],
        )
        from_transducer.assert_called_once()
        self.assertEqual(from_transducer.call_args.kwargs["num_threads"], 2)


if __name__ == "__main__":
    unittest.main()
