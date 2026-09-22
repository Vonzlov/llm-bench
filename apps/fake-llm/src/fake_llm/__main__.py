"""Запуск сервера: `python -m fake_llm` или команда `fake-llm`."""

import logging

import uvicorn

from fake_llm.app import create_app
from fake_llm.settings import Settings

logger = logging.getLogger("fake_llm")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings()
    logger.info("starting fake-llm with settings: %s", settings.model_dump_json())
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
