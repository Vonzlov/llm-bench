"""Схема запроса /v1/chat/completions — ровно то подмножество OpenAI API, которое нам нужно."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fake_llm.tokens import count_words


class ChatMessage(BaseModel):
    role: str
    # OpenAI допускает строку или список частей вида {"type": "text", "text": "..."}.
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if self.content is None:
            return ""
        return " ".join(str(part.get("text", "")) for part in self.content)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    # Остальные параметры OpenAI (temperature, seed, response_format…) принимаем и игнорируем.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    def token_limit(self) -> int | None:
        # max_completion_tokens — новое имя параметра в OpenAI API, max_tokens — старое.
        return self.max_completion_tokens or self.max_tokens

    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    def prompt_tokens(self) -> int:
        return sum(count_words(message.text()) for message in self.messages)
