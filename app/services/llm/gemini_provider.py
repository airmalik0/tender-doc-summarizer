"""Провайдер Google Gemini."""

from __future__ import annotations

import time

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage, parse_json_payload
from app.services.llm.schema import to_strict_json_schema


class GeminiProvider(LLMProvider):
    """Извлечение через generate_content с response_schema."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        from google import genai

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY не задан")

        self._model = settings.gemini_model
        self._timeout_ms = int(settings.llm_timeout_s * 1000)
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

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
        started = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": to_strict_json_schema(schema_model),
                    "max_output_tokens": max_output_tokens,
                    "http_options": {"timeout": self._timeout_ms},
                },
            )
        except Exception as exc:  # SDK Google не даёт единой типизированной иерархии ошибок
            raise LLMError(
                "Gemini вернул ошибку при генерации ответа.",
                details={"reason": str(exc)[:300]},
            ) from exc

        text = response.text or ""
        if not text.strip():
            raise LLMError("Модель вернула пустой ответ.")

        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            data=parse_json_payload(text, self.name),
            usage=LLMUsage(
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
