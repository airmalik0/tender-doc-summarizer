"""Провайдер Anthropic Claude — основной режим работы сервиса."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage, parse_json_payload
from app.services.llm.schema import to_strict_json_schema

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Извлечение через Messages API в режиме строгого структурированного вывода."""

    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY не задан")

        self._model = settings.anthropic_model
        self._effort = settings.anthropic_effort
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    @property
    def model(self) -> str | None:
        return self._model

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        max_output_tokens: int = 16_000,
    ) -> LLMResult:
        import anthropic

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_output_tokens,
                # Системный промпт стабилен от вызова к вызову, поэтому помечаем
                # его к кэшированию: при разборе многостраничного документа это
                # снимает повторную оплату одного и того же префикса.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self._effort,
                    "format": {
                        "type": "json_schema",
                        "schema": to_strict_json_schema(schema_model),
                    },
                },
                thinking={"type": "adaptive"},
            )
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Anthropic вернул ошибку {exc.status_code}.",
                details={"status": exc.status_code, "reason": _short(str(exc))},
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(
                "Не удалось связаться с Anthropic — проверьте сеть и доступность API.",
                details={"reason": _short(str(exc))},
            ) from exc

        # На запрос могут ответить отказом: это HTTP 200 с особым stop_reason,
        # и без явной проверки код пошёл бы разбирать пустой content.
        if response.stop_reason == "refusal":
            details: dict[str, Any] = {}
            if getattr(response, "stop_details", None) is not None:
                details["category"] = getattr(response.stop_details, "category", None)
            raise LLMError(
                "Модель отклонила запрос. Попробуйте другую модель в ANTHROPIC_MODEL.",
                details=details,
            )
        if response.stop_reason == "max_tokens":
            raise LLMError(
                "Ответ модели не поместился в лимит токенов — увеличьте max_output_tokens "
                "или уменьшите CHUNK_CHARS.",
                details={"max_tokens": max_output_tokens},
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise LLMError(
                "Модель вернула пустой ответ.", details={"stop_reason": response.stop_reason}
            )

        return LLMResult(
            data=parse_json_payload(text, self.name),
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def aclose(self) -> None:
        await self._client.close()


def _short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "…"
