# Короткие команды для локальной работы. Полный список: make help

VENV := .venv
PY   := $(VENV)/bin/python
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help
help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Создать виртуальное окружение и поставить зависимости
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

.PHONY: run
run: ## Поднять сервис локально с автоперезагрузкой
	$(VENV)/bin/uvicorn app.main:app --reload --port $(PORT)

.PHONY: test
test: ## Прогнать тесты
	$(VENV)/bin/pytest -q

.PHONY: lint
lint: ## Проверить стиль и типы
	$(VENV)/bin/ruff check app tests
	$(VENV)/bin/black --check app tests
	$(VENV)/bin/mypy app

.PHONY: format
format: ## Отформатировать код
	$(VENV)/bin/black app tests scripts
	$(VENV)/bin/ruff check --fix app tests

.PHONY: samples
samples: ## Пересобрать демонстрационные PDF
	$(PY) scripts/make_samples.py

.PHONY: up
up: ## Собрать и поднять сервис в Docker
	docker compose up --build

.PHONY: down
down: ## Остановить контейнеры
	docker compose down

.PHONY: smoke
smoke: ## Проверить поднятый сервис: health и разбор демо-документа (PORT=... если порт другой)
	@curl -sf http://localhost:$(PORT)/api/v1/health > /dev/null 2>&1 || { \
		echo "На http://localhost:$(PORT) не отвечает сервис."; \
		echo "Поднимите его (make up или make run) либо укажите порт: PORT=8090 make smoke"; \
		exit 1; }
	@curl -sf http://localhost:$(PORT)/api/v1/health | $(PY) -m json.tool
	@curl -sf -X POST "http://localhost:$(PORT)/api/v1/summarize/sample?name=tender-44fz-remont-krovli.pdf" \
		| $(PY) -c "import json,sys; d=json.load(sys.stdin); print('достоверность:', d['confidence'], '| цена:', d['price']['amount'], '| требований:', len(d['requirements']), '| санкций:', len(d['penalties']))"
