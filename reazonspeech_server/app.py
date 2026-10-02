import asyncio
import io
import json
import logging
import wave

import numpy as np
from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)
RUNTIME_KEY = web.AppKey("runtime")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
WAV_SAMPLE_TYPES = {1: ("u1", 128, 128), 2: ("<i2", 0, 32768), 4: ("<i4", 0, 2**31)}


async def health(request):
    return web.json_response({"status": "ok", "model": request.app[RUNTIME_KEY].model_name})


async def websocket(request):
    runtime = request.app[RUNTIME_KEY]
    socket = web.WebSocketResponse(heartbeat=30)
    await socket.prepare(request)
    uid = ""
    try:
        first = await socket.receive()
        if first.type != WSMsgType.TEXT:
            await socket.send_json({"error": "initial JSON configuration is required"})
            await socket.close(code=1003, message=b"initial JSON required")
            return socket
        try:
            configuration = json.loads(first.data)
            if not isinstance(configuration, dict):
                raise ValueError("configuration must be a JSON object")
            uid = str(configuration.get("uid", ""))
            if configuration.get("task", "transcribe") != "transcribe":
                raise ValueError("only task=transcribe is supported")
        except (json.JSONDecodeError, ValueError) as error:
            await socket.send_json({"uid": uid, "error": str(error)})
            await socket.close(code=1003, message=b"invalid configuration")
            return socket

        segmenter = runtime.create_segmenter()
        logger.info("client connected uid=%s", uid)
        await socket.send_json(
            {"uid": uid, "message": "SERVER_READY", "backend": runtime.model_name}
        )

        async for message in socket:
            if message.type == WSMsgType.BINARY:
                if len(message.data) % 4:
                    await socket.send_json({"uid": uid, "error": "PCM payload must contain float32 samples"})
                    continue
                samples = np.frombuffer(message.data, dtype="<f4").astype(np.float32, copy=True)
                for speech, start, end in segmenter.accept(samples):
                    logger.info("speech detected uid=%s samples=%d", uid, len(speech))
                    try:
                        text = (await runtime.transcribe(speech)).strip()
                    except Exception:
                        logger.exception("transcription failed uid=%s", uid)
                        await socket.send_json({"uid": uid, "error": "transcription failed"})
                        continue
                    if text:
                        await socket.send_json(
                            {
                                "uid": uid,
                                "segments": [
                                    {
                                        "text": text,
                                        "completed": True,
                                        "start": round(start, 3),
                                        "end": round(end, 3),
                                    }
                                ],
                            }
                        )
                        logger.info("transcription completed uid=%s", uid)
            elif message.type == WSMsgType.ERROR:
                logger.warning("websocket error uid=%s error=%s", uid, socket.exception())
                break
    except (ConnectionError, asyncio.CancelledError):
        raise
    finally:
        logger.info("client disconnected uid=%s", uid)
    return socket


def decode_wav(data):
    try:
        with wave.open(io.BytesIO(data)) as wav:
            width, channels, rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError(f"unsupported audio file, PCM WAV is required: {error}") from error
    if width not in WAV_SAMPLE_TYPES:
        raise ValueError("only 8/16/32-bit PCM WAV is supported")
    dtype, offset, scale = WAV_SAMPLE_TYPES[width]
    samples = (np.frombuffer(frames, dtype=dtype).astype(np.float32) - offset) / scale
    samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        # ponytail: linear-interpolation resampling, swap in a polyphase resampler if accuracy suffers
        samples = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples)
    return samples.astype(np.float32)


def split_speech(segmenter, samples):
    return segmenter.accept(samples) + segmenter.flush()


def openai_error(status, message):
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error"}}, status=status
    )


async def transcriptions(request):
    runtime = request.app[RUNTIME_KEY]
    form = await request.post()
    upload = form.get("file")
    if not isinstance(upload, web.FileField):
        return openai_error(400, "file is required")
    response_format = form.get("response_format", "json")
    if response_format not in ("json", "text", "verbose_json"):
        return openai_error(400, "response_format must be json, text or verbose_json")
    try:
        samples = decode_wav(upload.file.read())
    except ValueError as error:
        return openai_error(400, str(error))

    segments = []
    for speech, start, end in await asyncio.to_thread(split_speech, runtime.create_segmenter(), samples):
        try:
            text = (await runtime.transcribe(speech)).strip()
        except Exception:
            logger.exception("transcription failed file=%s", upload.filename)
            return openai_error(500, "transcription failed")
        if text:
            segments.append(
                {"id": len(segments), "start": round(start, 3), "end": round(end, 3), "text": text}
            )
    text = "".join(segment["text"] for segment in segments)
    logger.info("file transcription completed file=%s segments=%d", upload.filename, len(segments))

    if response_format == "text":
        return web.Response(text=text)
    if response_format == "verbose_json":
        return web.json_response(
            {
                "task": "transcribe",
                "language": "japanese",
                "duration": round(len(samples) / 16000, 3),
                "text": text,
                "segments": segments,
            }
        )
    return web.json_response({"text": text})


def create_app(runtime):
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    app[RUNTIME_KEY] = runtime
    app.router.add_get("/health", health)
    app.router.add_post("/v1/audio/transcriptions", transcriptions)
    app.router.add_get("/", websocket)
    app.router.add_get("/ws", websocket)
    return app
