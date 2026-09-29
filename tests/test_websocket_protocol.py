import asyncio
import struct
import unittest

from aiohttp.test_utils import TestClient, TestServer

from reazonspeech_server.app import create_app


class FakeSegmenter:
    def __init__(self):
        self.samples = []

    def accept(self, samples):
        self.samples.extend(samples.tolist())
        if len(self.samples) >= 4:
            result = self.samples[:]
            self.samples.clear()
            return [(result, 1.25, 1.25 + len(result) / 16000)]
        return []


class FakeRuntime:
    model_name = "reazonspeech-k2-v2"

    def __init__(self):
        self.segmenters = []
        self.seen_lengths = []

    def create_segmenter(self):
        segmenter = FakeSegmenter()
        self.segmenters.append(segmenter)
        return segmenter

    async def transcribe(self, samples):
        self.seen_lengths.append(len(samples))
        await asyncio.sleep(0)
        return f"recognized-{len(samples)}"


class WebSocketProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = FakeRuntime()
        self.client = TestClient(TestServer(create_app(self.runtime)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_health_reports_ready_model(self):
        response = await self.client.get("/health")

        self.assertEqual(response.status, 200)
        self.assertEqual(
            await response.json(),
            {"status": "ok", "model": "reazonspeech-k2-v2"},
        )

    async def test_client_receives_ready_then_completed_segment(self):
        socket = await self.client.ws_connect("/ws")
        await socket.send_json({"uid": "client-1", "task": "transcribe"})

        ready = await socket.receive_json()
        self.assertEqual(ready["uid"], "client-1")
        self.assertEqual(ready["message"], "SERVER_READY")
        self.assertEqual(ready["backend"], "reazonspeech-k2-v2")

        await socket.send_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
        result = await socket.receive_json(timeout=2)
        self.assertEqual(result["uid"], "client-1")
        self.assertEqual(
            result["segments"],
            [{"text": "recognized-4", "completed": True, "start": 1.25, "end": 1.25}],
        )
        await socket.close()

    async def test_each_connection_keeps_its_own_audio_and_uid(self):
        first = await self.client.ws_connect("/ws")
        second = await self.client.ws_connect("/ws")
        await first.send_json({"uid": "client-a", "task": "transcribe"})
        await second.send_json({"uid": "client-b", "task": "transcribe"})
        await first.receive_json()
        await second.receive_json()

        await first.send_bytes(struct.pack("<2f", 0.1, 0.2))
        await second.send_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
        await asyncio.sleep(0.05)
        await first.send_bytes(struct.pack("<2f", 0.3, 0.4))

        first_result, second_result = await asyncio.gather(
            first.receive_json(timeout=2), second.receive_json(timeout=2)
        )
        self.assertEqual(first_result["uid"], "client-a")
        self.assertEqual(second_result["uid"], "client-b")
        self.assertEqual(len(self.runtime.segmenters), 2)
        self.assertCountEqual(self.runtime.seen_lengths, [4, 4])
        await first.close()
        await second.close()

    async def test_rejects_non_float32_aligned_audio_without_closing_session(self):
        socket = await self.client.ws_connect("/ws")
        await socket.send_json({"uid": "client-1", "task": "transcribe"})
        await socket.receive_json()

        await socket.send_bytes(b"\x00\x01\x02")
        error = await socket.receive_json(timeout=2)
        self.assertEqual(error["uid"], "client-1")
        self.assertIn("error", error)

        await socket.send_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
        result = await socket.receive_json(timeout=2)
        self.assertEqual(result["segments"][0]["completed"], True)
        await socket.close()


if __name__ == "__main__":
    unittest.main()
