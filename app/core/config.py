"""Конфигурация приложения.

Все настройки читаются из переменных окружения (или из файла .env) и
валидируются pydantic-settings — так неверное значение падает на старте
приложения, а не посреди обработки чужого документа.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["auto", "anthropic", "openai", "gemini", "offline"]
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """Настройки сервиса."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Приложение ──────────────────────────────────────────────────────────
    app_name: str = "Тендерный суммаризатор"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ── Выбор LLM-провайдера ────────────────────────────────────────────────
    llm_provider: ProviderName = "auto"
    llm_timeout_s: float = 180.0
    llm_max_retries: int = 2

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: EffortLevel = "medium"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"

    # ── Ограничения загрузки ────────────────────────────────────────────────
    max_upload_mb: int = Field(default=25, ge=1, le=200)
    max_pages: int = Field(default=400, ge=1, le=2000)

    # ── OCR ─────────────────────────────────────────────────────────────────
    ocr_enabled: bool = True
    ocr_lang: str = "rus+eng"
    ocr_dpi: int = Field(default=220, ge=72, le=600)
    ocr_min_chars_per_page: int = Field(default=120, ge=0)

    # ── Чанкинг ─────────────────────────────────────────────────────────────
    chunk_chars: int = Field(default=14_000, ge=1_000, le=100_000)
    chunk_overlap_chars: int = Field(default=800, ge=0, le=10_000)
    max_chunks: int = Field(default=12, ge=1, le=100)

    # ── Кэш ─────────────────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_dir: Path = Path("data/cache")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def configured_providers(self) -> list[str]:
        """Провайдеры, для которых задан непустой API-ключ."""
        found: list[str] = []
        if self.anthropic_api_key and self.anthropic_api_key.get_secret_value().strip():
            found.append("anthropic")
        if self.openai_api_key and self.openai_api_key.get_secret_value().strip():
            found.append("openai")
        if self.gemini_api_key and self.gemini_api_key.get_secret_value().strip():
            found.append("gemini")
        return found


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон настроек (кэшируется на процесс)."""
    return Settings()
