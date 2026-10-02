import io
import unittest
import wave

import aiohttp
import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from reazonspeech_server.app import create_app


class FakeSegmenter:
    def accept(self, samples):
        self.samples = samples
        return [(samples[:4], 0.5, 1.0)]

    def flush(self):
        return [(self.samples[4:], 1.5, 2.0)]


class FakeRuntime:
    model_name = "reazonspeech-k2-v2"

    def __init__(self):
        self.seen = []

    def create_segmenter(self):
        return FakeSegmenter()

    async def transcribe(self, samples):
        self.seen.append(samples)
        return f"ことば{len(samples)}"


def wav_bytes(samples, rate=16000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


class TranscriptionsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = FakeRuntime()
        self.client = TestClient(TestServer(create_app(self.runtime)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def post(self, audio, **fields):
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="a.wav", content_type="audio/wav")
        form.add_field("model", "whisper-1")
        for key, value in fields.items():
            form.add_field(key, value)
        return await self.client.post("/v1/audio/transcriptions", data=form)

    async def test_json_joins_segments_including_flushed_tail(self):
        response = await self.post(wav_bytes([16384] * 10))

        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {"text": "ことば4ことば6"})
        np.testing.assert_allclose(self.runtime.seen[0], [0.5] * 4)

    async def test_verbose_json_reports_segments_and_duration(self):
        response = await self.post(wav_bytes([0] * 16000), response_format="verbose_json")

        body = await response.json()
        self.assertEqual(body["duration"], 1.0)
        self.assertEqual(
            body["segments"],
            [
                {"id": 0, "start": 0.5, "end": 1.0, "text": "ことば4"},
                {"id": 1, "start": 1.5, "end": 2.0, "text": "ことば15996"},
            ],
        )

    async def test_text_format_and_stereo_resampled_to_mono_16k(self):
        stereo_8k = np.repeat(np.arange(8000, dtype=np.int16), 2)
        response = await self.post(wav_bytes(stereo_8k, rate=8000, channels=2), response_format="text")

        self.assertEqual(await response.text(), "ことば4ことば15996")

    async def test_rejects_non_wav_and_missing_file(self):
        response = await self.post(b"ID3 not a wav")
        self.assertEqual(response.status, 400)
        self.assertIn("message", (await response.json())["error"])

        response = await self.client.post("/v1/audio/transcriptions", data={"model": "x"})
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
