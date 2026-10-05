# Единая точка входа: если команду приходится вспоминать, ей место здесь.
.PHONY: install lint format typecheck test check fake-llm probe load image-fake-llm

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

# Один запрос к модели с замерами. По умолчанию — в фейковую модель из make fake-llm.
probe:
	uv run bench-probe

# Быстрая ступенчатая нагрузка на фейковую модель: 4 ступени по 12 секунд.
# Полный прогон по методике — uv run bench-load без аргументов, около 14 минут.
load:
	uv run bench-load --levels 1,2,4,8 --warmup 2 --measure 10 --max-tokens 32

image-fake-llm:
	docker build -f apps/fake-llm/Dockerfile -t llm-bench/fake-llm:dev .
