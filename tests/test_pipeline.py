"""Тесты пайплайна.

Проверяется не «код отработал без исключения», а обещания сервиса:
выдуманная цитата не проходит проверку, сумма сверяется с независимым
парсером, отказ на одном фрагменте не роняет весь разбор.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.services.cache import SummaryCache
from app.services.extraction.pipeline import SummarizationPipeline
from app.services.llm.offline import OfflineProvider
from tests.conftest import ScriptedProvider, chunk_payload, evidence, money, synthesis_payload

REAL_QUOTE = "Начальная (максимальная) цена контракта составляет 12 480 350,00"
FAKE_QUOTE = "Начальная цена контракта составляет 99 999 999,00 рублей, чего в документе нет"


def _pipeline(provider, settings: Settings, tmp_path: Path) -> SummarizationPipeline:
    return SummarizationPipeline(
        settings, provider, SummaryCache(tmp_path / "cache", settings.cache_enabled)
    )


async def test_verified_quote_and_rule_confirmation(roof_pdf: bytes, settings, tmp_path) -> None:
    provider = ScriptedProvider(
        [chunk_payload(price=money(12480350.0, "12 480 350,00 руб.", evidence(1, REAL_QUOTE)))]
    )
    result = await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")

    assert result.price is not None
    assert result.price.evidence is not None
    assert result.price.evidence.verified is True
    assert result.price.confirmed_by_rules is True


async def test_fabricated_quote_is_flagged(roof_pdf: bytes, settings, tmp_path) -> None:
    """Модель назвала сумму, которой в документе нет, и приложила выдуманную цитату."""
    provider = ScriptedProvider(
        [chunk_payload(price=money(99999999.0, "99 999 999,00 руб.", evidence(1, FAKE_QUOTE)))]
    )
    result = await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")

    assert result.price is not None
    assert result.price.evidence is not None
    assert result.price.evidence.verified is False
    assert result.price.confirmed_by_rules is False
    assert any("не подтверждена детерминированным парсером" in w for w in result.warnings)
    assert any("Цитат не найдено" in w for w in result.warnings)


async def test_confidence_drops_on_unverified_evidence(roof_pdf: bytes, settings, tmp_path) -> None:
    good = ScriptedProvider(
        [chunk_payload(price=money(12480350.0, "12 480 350,00 руб.", evidence(1, REAL_QUOTE)))]
    )
    bad = ScriptedProvider(
        [chunk_payload(price=money(99999999.0, "99 999 999,00 руб.", evidence(1, FAKE_QUOTE)))]
    )

    high = await _pipeline(good, settings, tmp_path).run(data=roof_pdf, filename="a.pdf")
    low = await _pipeline(bad, settings, tmp_path).run(data=roof_pdf, filename="b.pdf")

    assert high.confidence > low.confidence


async def test_page_number_is_corrected(roof_pdf: bytes, settings, tmp_path) -> None:
    """Цитата настоящая, но страница названа неверно — номер чинится, факт остаётся."""
    provider = ScriptedProvider(
        [chunk_payload(price=money(12480350.0, "12 480 350,00", evidence(4, REAL_QUOTE)))]
    )
    result = await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")

    assert result.price.evidence.verified is True
    assert result.price.evidence.page == 1
    assert any("Номер страницы исправлен" in w for w in result.warnings)


async def test_failed_chunk_does_not_kill_the_run(roof_pdf: bytes, tmp_path) -> None:
    settings = Settings(
        chunk_chars=3000, chunk_overlap_chars=0, ocr_enabled=False, cache_enabled=False
    )
    provider = ScriptedProvider(
        [chunk_payload(price=money(12480350.0, "12 480 350,00", evidence(1, REAL_QUOTE)))],
        fail_on={0},
    )
    result = await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")

    assert result.meta.chunks > 1
    assert result.price is not None, "разбор должен собраться из уцелевших фрагментов"
    assert any("не разобран" in w for w in result.warnings)


async def test_all_chunks_failing_raises(roof_pdf: bytes, settings, tmp_path) -> None:
    provider = ScriptedProvider([chunk_payload()], fail_on={0, 1, 2, 3})

    with pytest.raises(LLMError):
        await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")


async def test_synthesis_failure_falls_back_to_template(
    roof_pdf: bytes, settings, tmp_path
) -> None:
    class BrokenSynthesis(ScriptedProvider):
        async def complete_json(self, *, system, user, schema_model, max_output_tokens=16_000):
            if schema_model.__name__ == "DocumentSynthesis":
                raise LLMError("Синтез не удался")
            return await super().complete_json(
                system=system,
                user=user,
                schema_model=schema_model,
                max_output_tokens=max_output_tokens,
            )

    provider = BrokenSynthesis(
        [
            chunk_payload(
                subject="Ремонт кровли",
                price=money(12480350.0, "12 480 350,00", evidence(1, REAL_QUOTE)),
            )
        ]
    )
    result = await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")

    assert result.summary, "факты уже извлечены, терять их из-за пересказа нельзя"
    assert any("по шаблону" in w for w in result.warnings)


async def test_offline_provider_is_capped_and_flagged(roof_pdf: bytes, settings, tmp_path) -> None:
    result = await _pipeline(OfflineProvider(), settings, tmp_path).run(
        data=roof_pdf, filename="t.pdf"
    )

    assert result.confidence <= 0.6
    assert result.warnings[0].startswith("Разбор выполнен без LLM")
    assert result.price is not None and result.price.amount == 12480350


async def test_cache_returns_previous_result(roof_pdf: bytes, tmp_path) -> None:
    settings = Settings(cache_enabled=True, ocr_enabled=False, cache_dir=tmp_path / "cache")
    provider = ScriptedProvider([chunk_payload(subject="Ремонт кровли")])
    pipeline = _pipeline(provider, settings, tmp_path)

    first = await pipeline.run(data=roof_pdf, filename="t.pdf")
    calls_after_first = provider.calls
    second = await pipeline.run(data=roof_pdf, filename="t.pdf")

    assert first.meta.cached is False
    assert second.meta.cached is True
    assert provider.calls == calls_after_first, "повторный разбор не должен ходить в модель"
    assert second.subject == first.subject


async def test_cache_can_be_bypassed(roof_pdf: bytes, tmp_path) -> None:
    settings = Settings(cache_enabled=True, ocr_enabled=False, cache_dir=tmp_path / "cache")
    provider = ScriptedProvider([chunk_payload()])
    pipeline = _pipeline(provider, settings, tmp_path)

    await pipeline.run(data=roof_pdf, filename="t.pdf")
    calls = provider.calls
    await pipeline.run(data=roof_pdf, filename="t.pdf", use_cache=False)

    assert provider.calls > calls


async def test_metrics_are_collected(roof_pdf: bytes, settings, tmp_path) -> None:
    provider = ScriptedProvider([chunk_payload()], synthesis=synthesis_payload("Выжимка", ["Риск"]))
    result = await _pipeline(provider, settings, tmp_path).run(data=roof_pdf, filename="t.pdf")

    assert result.meta.provider == "scripted"
    assert result.meta.llm_calls == 2  # извлечение + синтез
    assert result.meta.input_tokens == 110
    assert result.meta.pdf_ms >= 0
    assert result.document.sha256 and len(result.document.sha256) == 64
