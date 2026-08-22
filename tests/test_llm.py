"""Тесты слоя провайдеров: выбор, разбор ответа, offline-режим."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMResponseInvalidError, ProviderNotConfiguredError
from app.models.extraction import ChunkExtraction
from app.services.llm.base import parse_json_payload
from app.services.llm.factory import build_provider, resolve_provider_name
from app.services.llm.offline import OfflineProvider, split_pages


def test_auto_without_keys_falls_back_to_offline() -> None:
    settings = Settings(llm_provider="auto")
    assert resolve_provider_name(settings) == "offline"
    assert build_provider(settings).name == "offline"


def test_auto_prefers_anthropic() -> None:
    settings = Settings(llm_provider="auto", anthropic_api_key="test", openai_api_key="test")
    assert resolve_provider_name(settings) == "anthropic"


def test_auto_uses_the_only_configured_provider() -> None:
    assert resolve_provider_name(Settings(llm_provider="auto", gemini_api_key="test")) == "gemini"


def test_explicit_provider_without_key_is_an_error() -> None:
    """Явно названный провайдер обязан подняться — молчаливое понижение режима хуже ошибки."""
    with pytest.raises(ProviderNotConfiguredError):
        build_provider(Settings(llm_provider="openai"))


def test_blank_key_counts_as_missing() -> None:
    assert Settings(anthropic_api_key="   ").configured_providers() == []


def test_parse_json_payload_reads_plain_json() -> None:
    assert parse_json_payload('{"a": 1}', "test") == {"a": 1}


def test_parse_json_payload_unwraps_markdown_fence() -> None:
    assert parse_json_payload('```json\n{"a": 1}\n```', "test") == {"a": 1}


def test_parse_json_payload_rejects_garbage() -> None:
    with pytest.raises(LLMResponseInvalidError):
        parse_json_payload("совсем не json", "test")


def test_parse_json_payload_rejects_non_object() -> None:
    with pytest.raises(LLMResponseInvalidError):
        parse_json_payload("[1, 2, 3]", "test")


def test_split_pages_restores_markers() -> None:
    pages = split_pages("[СТРАНИЦА 1]\nпервая\n\n[СТРАНИЦА 2]\nвторая")
    assert pages == [(1, "первая"), (2, "вторая")]


def test_split_pages_without_markers_is_one_page() -> None:
    assert split_pages("просто текст") == [(1, "просто текст")]


async def test_offline_provider_matches_the_schema(roof_pdf: bytes, settings) -> None:
    """Offline-провайдер обязан выдавать ровно тот же контракт, что и модель."""
    from app.services.pdf.extractor import extract_document

    document = extract_document(roof_pdf, settings)
    result = await OfflineProvider().complete_json(
        system="", user=document.as_prompt_text(), schema_model=ChunkExtraction
    )
    extraction = ChunkExtraction.model_validate(result.data)

    assert extraction.price.amount == 12480350.0
    assert extraction.requirements
    assert extraction.penalties


async def test_offline_quotes_are_taken_from_the_document(roof_pdf: bytes, settings) -> None:
    """Побочный эффект правил: цитата вырезана из текста, значит подтвердится всегда."""
    from app.services.extraction.verify import DocumentIndex
    from app.services.pdf.extractor import extract_document

    document = extract_document(roof_pdf, settings)
    index = DocumentIndex([(page.number, page.text) for page in document.pages])
    result = await OfflineProvider().complete_json(
        system="", user=document.as_prompt_text(), schema_model=ChunkExtraction
    )
    extraction = ChunkExtraction.model_validate(result.data)

    for requirement in extraction.requirements:
        assert index.check(requirement.evidence.quote, requirement.evidence.page).verified


def test_offline_provider_cannot_synthesize() -> None:
    assert OfflineProvider().supports_synthesis is False


def test_prompt_artefacts_do_not_leak_into_pages() -> None:
    """Закрывающий тег промпта и пометка о перекрытии — не часть документа."""
    prompt = "[СТРАНИЦА 1]\n(продолжение предыдущего фрагмента)\nтекст документа\n</документ>"
    pages = split_pages(prompt)
    assert pages == [(1, "текст документа")]


def test_gemini_schema_has_no_additional_properties() -> None:
    """Бекенд Google отвергает запрос со словом additionalProperties целиком."""
    import json

    from app.models.extraction import ChunkExtraction
    from app.services.llm.gemini_provider import strip_unsupported
    from app.services.llm.schema import to_strict_json_schema

    cleaned = json.dumps(strip_unsupported(to_strict_json_schema(ChunkExtraction)))
    assert "additionalProperties" not in cleaned
    assert "properties" in cleaned


def test_openai_and_anthropic_schema_keeps_additional_properties() -> None:
    """А им это поле, наоборот, обязательно для строгого режима."""
    import json

    from app.models.extraction import ChunkExtraction
    from app.services.llm.schema import to_strict_json_schema

    assert "additionalProperties" in json.dumps(to_strict_json_schema(ChunkExtraction))
