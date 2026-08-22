"""Провайдер OpenAI.

Реализован ради требования ТЗ «любая доступная LLM» и как проверка того,
что контракт провайдера действительно провайдеро-независим: та же схема,
тот же промпт, другой SDK.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage, parse_json_payload
from app.services.llm.schema import to_strict_json_schema


class OpenAIProvider(LLMProvider):
    """Извлечение через Chat Completions с response_format = json_schema."""

    name = "openai"

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY не задан")

        self._model = settings.openai_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
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
        import openai

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=max_output_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_model.__name__,
                        "strict": True,
                        "schema": to_strict_json_schema(schema_model),
                    },
                },
            )
        except openai.APIStatusError as exc:
            raise LLMError(
                f"OpenAI вернул ошибку {exc.status_code}.",
                details={"status": exc.status_code, "reason": str(exc)[:300]},
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(
                "Не удалось связаться с OpenAI — проверьте сеть и доступность API.",
                details={"reason": str(exc)[:300]},
            ) from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(
                "Ответ модели не поместился в лимит токенов — уменьшите CHUNK_CHARS.",
                details={"max_tokens": max_output_tokens},
            )
        content = choice.message.content or ""
        if not content.strip():
            raise LLMError(
                "Модель вернула пустой ответ.", details={"finish_reason": choice.finish_reason}
            )

        usage = response.usage
        return LLMResult(
            data=parse_json_payload(content, self.name),
            usage=LLMUsage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def aclose(self) -> None:
        await self._client.close()
