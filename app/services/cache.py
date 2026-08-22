"""Кэш разборов по содержимому файла.

Разбор одного документа стоит реальных денег и десятков секунд, а
проверяющий почти наверняка загрузит один и тот же файл не один раз.
Ключ — хэш содержимого вместе с провайдером, моделью и версией схемы:
сменилась модель или формат ответа — прежние записи просто перестают
находиться, чистить ничего не нужно.

Хранилище файловое, по одному JSON на разбор. Redis тут был бы лишней
инфраструктурой: данные локальные, объёмы крошечные, а требование «запустил
одной командой» дороже.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.core.logging import get_logger
from app.models.summary import TenderSummary

logger = get_logger(__name__)

# Меняется вместе с форматом TenderSummary — старые записи станут недоступны.
CACHE_VERSION = "v1"


def content_hash(data: bytes) -> str:
    """SHA-256 содержимого файла."""
    return hashlib.sha256(data).hexdigest()


class SummaryCache:
    """Файловый кэш готовых разборов."""

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self._directory = directory
        self._enabled = enabled
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def key(self, sha256: str, provider: str, model: str | None) -> str:
        return f"{CACHE_VERSION}-{provider}-{model or 'none'}-{sha256[:32]}"

    def get(self, key: str) -> TenderSummary | None:
        if not self._enabled:
            return None

        path = self._path(key)
        if not path.exists():
            return None

        try:
            return TenderSummary.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Битая или устаревшая запись не должна ломать запрос: просто чиним кэш.
            logger.warning("Запись кэша %s повреждена, удаляю: %s", key, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, summary: TenderSummary) -> None:
        if not self._enabled:
            return

        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Переименование атомарно: параллельный запрос не прочитает наполовину записанный файл.
            temporary.replace(path)
        except OSError as exc:  # pragma: no cover — зависит от файловой системы
            logger.warning("Не удалось записать кэш %s: %s", key, exc)
            temporary.unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        return self._directory / f"{key}.json"
