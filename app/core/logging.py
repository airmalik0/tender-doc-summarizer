"""Настройка логирования и сквозной идентификатор запроса.

Идентификатор запроса живёт в contextvar, поэтому попадает в каждую строку
лога, написанную во время обработки запроса, — включая логи из глубины
пайплайна, которым не нужно ничего знать про HTTP.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Подмешивает текущий request_id в каждую запись лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Конфигурирует корневой логгер один раз за жизнь процесса."""
    root = logging.getLogger()
    if getattr(root, "_tender_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn пишет свой access-лог в собственном формате — приглушаем дубли.
    logging.getLogger("uvicorn.access").propagate = False
    for noisy in ("httpx", "httpcore", "anthropic", "openai", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._tender_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
