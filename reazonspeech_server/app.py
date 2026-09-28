import asyncio
import json
import logging

import numpy as np
from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)
RUNTIME_KEY = web.AppKey("runtime")


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


def create_app(runtime):
    app = web.Application()
    app[RUNTIME_KEY] = runtime
    app.router.add_get("/health", health)
    app.router.add_get("/", websocket)
    app.router.add_get("/ws", websocket)
    return app
