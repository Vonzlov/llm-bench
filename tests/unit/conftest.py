"""Фикстура с фейковой моделью — общая для её собственных тестов и для тестов ядра.

Сервер поднимается настоящим uvicorn на свободном порту, и тесты ходят в него по
HTTP — так же, как раннер и генератор нагрузки ходят в настоящие модели. Поэтому
время в тестах измеряется вместе с реальными накладными расходами сети.
"""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import uvicorn

from fake_llm.app import create_app
from fake_llm.settings import Settings
from tests.unit.fake_llm.helpers import StartServer


@contextmanager
def running_server(settings: Settings) -> Iterator[str]:
    # IPPROTO_TCP указываем явно: с proto=0 asyncio не включает TCP_NODELAY на принятых
    # соединениях, и алгоритм Нагла вместе с отложенным ACK добавляет ~40 мс к каждому ответу.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("fake-llm did not start within 10 seconds")
        time.sleep(0.01)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.fixture
def start_server() -> Iterator[StartServer]:
    """Запускает фейковую модель с нужными настройками и возвращает её адрес.

    Пример: base_url = start_server(ttft_ms=100, output_tokens=5).
    Все серверы, запущенные в тесте, останавливаются после него.
    """
    with ExitStack() as stack:

        def start(**overrides: Any) -> str:
            return stack.enter_context(running_server(Settings(**overrides)))

        yield start
