# Образ фейковой модели. Собирается из корня репозитория, потому что uv.lock общий:
#   docker build -f apps/fake-llm/Dockerfile -t llm-bench/fake-llm:dev .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /bin/uv

# Байткод компилируем при сборке: в контейнере код работает от непривилегированного пользователя.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Слой с зависимостями пересобирается, только когда меняются lock-файл или pyproject.
COPY pyproject.toml uv.lock ./
COPY apps/fake-llm/pyproject.toml apps/fake-llm/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package fake-llm --no-install-workspace

COPY apps/fake-llm apps/fake-llm
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package fake-llm

ENV PATH="/app/.venv/bin:$PATH"
USER 65534
EXPOSE 8000
CMD ["fake-llm"]
