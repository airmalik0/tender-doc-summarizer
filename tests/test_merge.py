"""Тесты слияния фрагментов."""

from __future__ import annotations

from app.models.extraction import ChunkExtraction
from app.services.extraction.merge import merge_chunks
from tests.conftest import chunk_payload, date_value, evidence, money


def _chunk(**overrides) -> ChunkExtraction:
    return ChunkExtraction.model_validate(chunk_payload(**overrides))


def test_empty_input_yields_empty_facts() -> None:
    facts = merge_chunks([])
    assert facts.price is None
    assert facts.requirements == []


def test_sentinels_do_not_become_values() -> None:
    """Пустая сумма из схемы не должна превратиться в «цену контракта, равную нулю»."""
    facts = merge_chunks([_chunk(), _chunk()])
    assert facts.price is None
    assert facts.application_deadline is None


def test_most_frequent_price_wins_and_conflict_is_reported() -> None:
    facts = merge_chunks(
        [
            _chunk(price=money(12480350.0, "12 480 350,00 руб.")),
            _chunk(price=money(12480350.0, "12 480 350,00 руб.")),
            _chunk(price=money(999.0, "999 руб.")),
        ]
    )

    assert facts.price is not None
    assert facts.price.amount == 12480350.0
    assert facts.conflicts, "расхождение обязано попасть в конфликты, а не потеряться"
    assert "цену контракта" in facts.conflicts[0]


def test_single_price_produces_no_conflict() -> None:
    facts = merge_chunks([_chunk(price=money(100.0, "100 руб.")), _chunk()])
    assert facts.conflicts == []


def test_duplicate_requirements_are_collapsed() -> None:
    text = "Наличие членства в СРО в области строительства с взносом в компенсационный фонд"
    facts = merge_chunks(
        [
            _chunk(requirements=[_requirement(text)]),
            _chunk(requirements=[_requirement(text)]),
        ]
    )
    assert len(facts.requirements) == 1


def test_truncated_duplicate_keeps_longer_wording() -> None:
    """На границе перекрытия требование приезжает обрезанным."""
    full = "Наличие в штате не менее 2 специалистов по организации строительства из НРС"
    cut = "Наличие в штате не менее 2 специалистов по организации"
    facts = merge_chunks(
        [_chunk(requirements=[_requirement(cut)]), _chunk(requirements=[_requirement(full)])]
    )

    assert len(facts.requirements) == 1
    assert facts.requirements[0].text == full


def test_distinct_requirements_are_kept() -> None:
    facts = merge_chunks(
        [
            _chunk(requirements=[_requirement("Членство в СРО в области строительства")]),
            _chunk(requirements=[_requirement("Гарантийный срок на оборудование 24 месяца")]),
        ]
    )
    assert len(facts.requirements) == 2


def test_longest_subject_wins_on_tie() -> None:
    facts = merge_chunks(
        [_chunk(subject="Ремонт кровли"), _chunk(subject="Капитальный ремонт кровли школы")]
    )
    assert facts.subject == "Капитальный ремонт кровли школы"


def test_undetermined_law_is_ignored() -> None:
    facts = merge_chunks(
        [_chunk(law="не определено"), _chunk(law="44-ФЗ"), _chunk(law="не определено")]
    )
    assert facts.law == "44-ФЗ"


def test_date_conflict_is_reported() -> None:
    facts = merge_chunks(
        [
            _chunk(application_deadline=date_value("2026-09-15", "15 сентября")),
            _chunk(application_deadline=date_value("2026-10-01", "1 октября")),
        ]
    )
    assert facts.application_deadline is not None
    assert any("подачи заявок" in message for message in facts.conflicts)


def _requirement(text: str) -> dict:
    return {
        "text": text,
        "category": "прочее",
        "mandatory": True,
        "evidence": evidence(1, text),
    }
