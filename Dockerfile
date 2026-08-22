# Образ сервиса. Один stage: зависимости — колёса, компилировать нечего,
# а лишний stage только запутал бы сборку.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# tesseract и русский языковой пакет: без них не работает OCR-ветка, а
# документы с площадок регулярно приходят сканами. eng идёт в базовом
# пакете и нужен для латиницы в номерах и артикулах.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем: правка кода не должна тянуть за собой
# переустановку всего окружения.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY samples ./samples

# Работаем не из-под root: контейнеру нужны только чтение кода и запись
# в каталог кэша.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Проверка живости ходит в тот же эндпоинт, что и человек, — так падение
# приложения видно оркестратору, а не только в логах.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
