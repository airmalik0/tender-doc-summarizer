"""Единый формат ошибок.

Клиенту (в том числе нашему же веб-интерфейсу) нужен предсказуемый ответ:
машиночитаемый код, человекочитаемое сообщение и идентификатор запроса,
по которому ошибку можно найти в логах. Никаких HTML-страниц Starlette и
никаких голых строк.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger, request_id_var
from app.models.summary import ErrorResponse

logger = get_logger(__name__)


def _response(
    status_code: int, code: str, message: str, details: dict | None = None
) -> JSONResponse:
    payload = ErrorResponse(
        code=code,
        message=message,
        details=details or {},
        request_id=request_id_var.get(),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def register_error_handlers(app: FastAPI) -> None:
    """Вешает обработчики на приложение."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        # Ожидаемая ошибка: логируем предупреждением, без трассировки.
        logger.warning("%s: %s", exc.code, exc.message)
        return _response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Код указан числом, а не константой Starlette: её имя менялось между
        # версиями, и привязываться к нему в обработчике ошибок незачем.
        return _response(
            422,
            "validation_error",
            "Запрос не прошёл валидацию.",
            {"errors": exc.errors()[:5]},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _response(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Неожиданная ошибка: полная трассировка в лог, наружу — только код.
        logger.exception("Необработанная ошибка: %s", exc)
        return _response(
            500,
            "internal_error",
            "Внутренняя ошибка сервиса. Подробности — в логах по request_id.",
        )
