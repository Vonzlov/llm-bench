# Единая точка входа: если команду приходится вспоминать, ей место здесь.
.PHONY: install lint format typecheck test check fake-llm image-fake-llm

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

# То же, что проверяет CI.
check: lint typecheck test

# Фейковая модель на http://localhost:8000, настройки — через переменные FAKE_LLM_*.
fake-llm:
	uv run fake-llm

image-fake-llm:
	docker build -f apps/fake-llm/Dockerfile -t llm-bench/fake-llm:dev .
