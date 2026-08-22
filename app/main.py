"""Точка входа сервиса.

Провайдер, кэш и пайплайн создаются один раз на старте: клиент провайдера
держит пул соединений, и пересоздавать его на каждый запрос — верный способ
упереться в лимиты на ровном месте.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.core.config import get_settings
from app.core.exceptions import ProviderNotConfiguredError
from app.core.logging import get_logger, request_id_var, setup_logging
from app.services.cache import SummaryCache
from app.services.extraction.pipeline import SummarizationPipeline
from app.services.llm.factory import build_provider

logger = get_logger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
# Запас на границы и заголовки multipart поверх самого файла.
_MULTIPART_OVERHEAD = 8192

DESCRIPTION = """\
Сервис извлекает из документации госзакупок структурированную выжимку:
сумму контракта, сроки, ключевые требования к исполнителю и меры ответственности.

Каждое извлечённое значение сопровождается цитатой из документа с номером
страницы, а цитата проверяется поиском по исходному тексту — поле `verified`
показывает результат проверки. Ключевые суммы дополнительно сверяются с
детерминированным парсером: `confirmed_by_rules`.

Интерактивный интерфейс — на `/`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)

    try:
        provider = build_provider(settings)
    except ProviderNotConfiguredError as exc:
        # Не роняем приложение трассировкой: конфигурационная ошибка должна
        # читаться в логе одной строкой, а не тонуть в стеке вызовов.
        logger.error("Сервис не запущен. %s", exc.message)
        raise SystemExit(1) from None
    cache = SummaryCache(settings.cache_dir, settings.cache_enabled)

    app.state.settings = settings
    app.state.provider = provider
    app.state.pipeline = SummarizationPipeline(settings, provider, cache)

    logger.info(
        "%s %s запущен: провайдер %s, модель %s",
        settings.app_name,
        settings.app_version,
        provider.name,
        provider.model or "—",
    )
    try:
        yield
    finally:
        await provider.aclose()
        logger.info("Сервис остановлен")


def create_app() -> FastAPI:
    """Фабрика приложения — так его удобно поднимать в тестах."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # Сервис локальный и односерверный: интерфейс лежит на том же адресе.
    # Разрешение любых источников нужно, чтобы разбор можно было дёрнуть
    # из чужой страницы или из curl без плясок с заголовками.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def reject_oversized_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Отказ по Content-Length до разбора тела запроса.

        Без этого FastAPI сначала полностью принимает multipart (крупная часть
        уходит во временный файл на диске) и только потом обработчик смотрит на
        размер. Заголовку можно не поверить — он не обязателен и его можно
        подделать, — поэтому проверка в обработчике остаётся. Но честную
        большую загрузку она отсекает сразу.
        """
        declared = request.headers.get("content-length")
        if declared and declared.isdigit():
            limit = settings.max_upload_bytes + _MULTIPART_OVERHEAD
            if int(declared) > limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "code": "file_too_large",
                        "message": (f"Тело запроса больше допустимых {settings.max_upload_mb} МБ."),
                        "details": {"limit_bytes": settings.max_upload_bytes},
                        "request_id": None,
                    },
                )
        return await call_next(request)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex[:8]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed = int((time.perf_counter() - started) * 1000)
            response.headers["X-Request-Id"] = request_id
            response.headers["X-Process-Time-Ms"] = str(elapsed)
            # Логируем и проставляем заголовок ДО сброса контекста: иначе и в
            # заголовке, и в строке лога окажется прочерк вместо идентификатора.
            if request.url.path.startswith("/api"):
                logger.info(
                    "%s %s → %s за %d мс",
                    request.method,
                    request.url.path,
                    response.status_code,
                    elapsed,
                )
            return response
        finally:
            request_id_var.reset(token)

    register_error_handlers(app)
    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> Response:
        page = WEB_DIR / "index.html"
        if not page.exists():  # pragma: no cover — только при поломанной сборке
            return JSONResponse({"message": "Интерфейс не найден. Документация API — на /docs."})
        return FileResponse(page)

    return app


app = create_app()
