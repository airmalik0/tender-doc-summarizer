"""Доменные исключения.

Каждое исключение несёт HTTP-статус и машиночитаемый код, чтобы обработчик
в app/api/errors.py мог отдать единообразный ответ об ошибке, не разбирая
текст сообщения.
"""

from __future__ import annotations


class AppError(Exception):
    """Базовая ошибка приложения, ожидаемая и обработанная."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class PdfParseError(AppError):
    status_code = 422
    code = "pdf_parse_error"


class EmptyDocumentError(AppError):
    status_code = 422
    code = "empty_document"


class DocumentTooLongError(AppError):
    status_code = 413
    code = "document_too_long"


class ProviderNotConfiguredError(AppError):
    status_code = 503
    code = "provider_not_configured"


class LLMError(AppError):
    """Провайдер не ответил или ответил ошибкой."""

    status_code = 502
    code = "llm_error"


class LLMResponseInvalidError(AppError):
    """Провайдер ответил, но структура ответа не прошла валидацию схемы."""

    status_code = 502
    code = "llm_response_invalid"
