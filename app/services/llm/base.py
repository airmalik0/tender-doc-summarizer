"""Общий контракт LLM-провайдера.

Пайплайн извлечения не должен знать, чей SDK стоит за вызовом. Ему нужно
одно: отдать системный промпт, текст фрагмента и схему — получить словарь,
соответствующий схеме, плюс расход токенов для метрик.

Отдельно вынесен признак supports_synthesis. Итоговую выжимку связным
текстом умеет писать только настоящая модель; offline-провайдер собирает
её по шаблону, и пайплайн должен уметь это различать явно, а не по
названию класса.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel

from app.core.exceptions import LLMResponseInvalidError

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*\})\s*```", re.DOTALL)


@dataclass(slots=True)
class LLMUsage:
    """Расход токенов на один вызов."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=_add_optional(self.input_tokens, other.input_tokens),
            output_tokens=_add_optional(self.output_tokens, other.output_tokens),
        )


def _add_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


@dataclass(slots=True)
class LLMResult:
    """Ответ провайдера: разобранный JSON и метрики вызова."""

    data: dict[str, Any]
    usage: LLMUsage
    latency_ms: int


class LLMProvider(ABC):
    """Провайдер структурированного извлечения."""

    name: ClassVar[str] = "unknown"
    supports_synthesis: ClassVar[bool] = True

    @property
    @abstractmethod
    def model(self) -> str | None:
        """Идентификатор модели для метрик и диагностики."""

    @abstractmethod
    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        max_output_tokens: int = 16_000,
    ) -> LLMResult:
        """Возвращает JSON-объект, соответствующий схеме schema_model."""

    async def aclose(self) -> None:
        """Освобождает ресурсы клиента. По умолчанию делать нечего."""
        return None


def parse_json_payload(text: str, provider: str) -> dict[str, Any]:
    """Разбирает JSON из ответа модели.

    В строгом режиме структурированного вывода ответ и так обязан быть чистым
    JSON, но провайдеры иногда оборачивают его в markdown-ограждение. Это
    дешевле пережить, чем ронять весь разбор документа.
    """
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        fenced = _JSON_FENCE_RE.search(stripped)
        if not fenced:
            raise LLMResponseInvalidError(
                f"Провайдер {provider} вернул ответ, который не разбирается как JSON.",
                details={"preview": stripped[:400]},
            ) from None
        try:
            payload = json.loads(fenced.group("body"))
        except json.JSONDecodeError as exc:
            raise LLMResponseInvalidError(
                f"Провайдер {provider} вернул повреждённый JSON.",
                details={"preview": stripped[:400], "reason": str(exc)},
            ) from exc

    if not isinstance(payload, dict):
        raise LLMResponseInvalidError(
            f"Провайдер {provider} вернул {type(payload).__name__} вместо объекта.",
            details={"preview": stripped[:400]},
        )
    return payload
