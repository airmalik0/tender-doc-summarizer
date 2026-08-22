"""Общие фикстуры.

Весь набор тестов выполняется без сети и без API-ключей: там, где нужен
провайдер, подставляется либо offline-провайдер на правилах, либо
сценарный, отвечающий заранее заданным JSON. Тест, которому нужен интернет,
рано или поздно станет тестом, который «иногда падает».
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

# Тесты не должны зависеть от .env разработчика: настоящий ключ в файле
# превратил бы «auto без ключей» в «auto с Anthropic» и уронил бы проверки
# выбора провайдера. Переменные окружения приоритетнее dotenv, поэтому
# гасим ключи явно.
os.environ["LLM_PROVIDER"] = "offline"
os.environ["CACHE_ENABLED"] = "false"
os.environ["OCR_ENABLED"] = "false"
for _key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
    os.environ[_key] = ""

from app.core.config import Settings, get_settings  # noqa: E402
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Настройки для теста: без сети, без кэша, без OCR."""
    return Settings(
        llm_provider="offline",
        cache_enabled=False,
        ocr_enabled=False,
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture(scope="session")
def roof_pdf() -> bytes:
    return (SAMPLES / "tender-44fz-remont-krovli.pdf").read_bytes()


@pytest.fixture(scope="session")
def supply_pdf() -> bytes:
    return (SAMPLES / "tender-44fz-postavka-oborudovaniya.pdf").read_bytes()


@pytest.fixture(scope="session")
def scan_pdf() -> bytes:
    return (SAMPLES / "tender-scan-postavka-kanctovarov.pdf").read_bytes()


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Настройки кэшируются на процесс — между тестами кэш сбрасывается."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def empty_evidence() -> dict[str, Any]:
    return {"page": 0, "quote": ""}


def evidence(page: int, quote: str) -> dict[str, Any]:
    return {"page": page, "quote": quote}


def money(amount: float = 0.0, raw: str = "", ev: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "amount": amount,
        "currency": "RUB" if amount else "",
        "vat": "не указано",
        "raw_text": raw,
        "evidence": ev or empty_evidence(),
    }


def date_value(iso: str = "", raw: str = "", ev: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"iso_date": iso, "raw_text": raw, "evidence": ev or empty_evidence()}


def chunk_payload(**overrides: Any) -> dict[str, Any]:
    """Валидный ответ модели на извлечение — база для подмены отдельных полей."""
    payload: dict[str, Any] = {
        "subject": "",
        "customer": "",
        "procurement_number": "",
        "law": "не определено",
        "price": money(),
        "contract_security": money(),
        "application_deadline": date_value(),
        "contract_start": date_value(),
        "contract_end": date_value(),
        "duration_text": "",
        "stages": [],
        "requirements": [],
        "penalties": [],
    }
    payload.update(overrides)
    return payload


def synthesis_payload(summary: str = "Краткая выжимка.", risks: list[str] | None = None):
    return {"summary": summary, "risks": risks if risks is not None else ["Риск"]}


class ScriptedProvider(LLMProvider):
    """Провайдер, отвечающий заранее заданными полезными нагрузками."""

    name = "scripted"

    def __init__(
        self,
        extractions: list[dict[str, Any]],
        synthesis: dict[str, Any] | None = None,
        fail_on: set[int] | None = None,
    ) -> None:
        self._extractions = list(extractions)
        self._synthesis = synthesis if synthesis is not None else synthesis_payload()
        self._fail_on = fail_on or set()
        self.calls = 0

    @property
    def model(self) -> str | None:
        return "scripted-1"

    async def complete_json(
        self, *, system: str, user: str, schema_model: type, max_output_tokens: int = 16_000
    ) -> LLMResult:
        from app.core.exceptions import LLMError

        index = self.calls
        self.calls += 1

        if schema_model.__name__ == "DocumentSynthesis":
            return LLMResult(data=dict(self._synthesis), usage=LLMUsage(10, 5), latency_ms=1)

        if index in self._fail_on:
            raise LLMError("Смоделированный отказ провайдера.")

        payload = self._extractions[min(index, len(self._extractions) - 1)]
        return LLMResult(data=dict(payload), usage=LLMUsage(100, 50), latency_ms=1)
