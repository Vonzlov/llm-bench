"""Настройки фейковой модели.

Всё читается из переменных окружения с префиксом FAKE_LLM_, например
FAKE_LLM_TTFT_MS=200 или FAKE_LLM_ERROR_RATE=0.05.
"""

from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FAKE_LLM_")

    # Имя, под которым модель видна в /v1/models. Запросы к другому имени получают 404, как у vLLM.
    served_model_name: str = "fake-llm"
    host: str = "0.0.0.0"
    port: int = 8000

    # Время до первого токена, мс. Ожидание свободного слота в очереди добавляется сверху.
    ttft_ms: float = Field(default=200.0, ge=0)
    # Скорость генерации после первого токена, токенов в секунду.
    tokens_per_second: float = Field(default=50.0, gt=0)
    # Длина ответа, если клиент не ограничил её через max_tokens.
    output_tokens: int = Field(default=64, ge=1)
    # Сколько запросов обрабатывается одновременно, остальные ждут в очереди.
    # Работает как OLLAMA_NUM_PARALLEL; 0 — без ограничения, как идеальный батчинг.
    max_concurrency: int = Field(default=0, ge=0)

    # Доли сбоев, от 0 до 1.
    error_rate: float = Field(default=0.0, ge=0, le=1)  # ответ 500
    rate_limit_rate: float = Field(default=0.0, ge=0, le=1)  # ответ 429 с Retry-After
    truncate_rate: float = Field(default=0.0, ge=0, le=1)  # стрим обрывается на середине
    # Зерно генератора случайных чисел: при том же порядке запросов сбои повторяются.
    seed: int | None = None

    @model_validator(mode="after")
    def check_error_rates(self) -> Self:
        # Обе ошибки разыгрываются одним броском, поэтому их доли в сумме не больше единицы.
        if self.error_rate + self.rate_limit_rate > 1:
            raise ValueError("error_rate + rate_limit_rate must not exceed 1")
        return self
