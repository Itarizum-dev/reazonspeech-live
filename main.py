import logging

from aiohttp import web

from reazonspeech_server.app import create_app
from reazonspeech_server.config import Settings
from reazonspeech_server.runtime import ReazonRuntime


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()
    runtime = ReazonRuntime.load(settings)
    logging.getLogger(__name__).info("WebSocket server started on %s:%d", settings.host, settings.port)
    web.run_app(create_app(runtime), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
