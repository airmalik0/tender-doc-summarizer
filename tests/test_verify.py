"""Тесты сверки цитат.

Здесь проверяется главное обещание сервиса: выдуманная цитата не проходит,
а настоящая — проходит, даже пережив OCR.
"""

from __future__ import annotations

import pytest

from app.services.extraction.verify import DocumentIndex, normalize

PAGES = [
    (
        1,
        "Начальная (максимальная) цена контракта составляет 12 480 350,00 "
        "(двенадцать миллионов четыреста восемьдесят тысяч триста пятьдесят) рублей "
        "00 копеек, в том числе НДС 20%.",
    ),
    (
        2,
        "За каждый факт неисполнения подрядчиком обязательства устанавливается штраф "
        "в размере 3 (трёх) процентов цены контракта, что составляет 374 410,50 рубля.",
    ),
]


@pytest.fixture
def index() -> DocumentIndex:
    return DocumentIndex(PAGES)


def test_normalize_collapses_typography() -> None:
    assert normalize("«Ёлка»  —\nтест") == '"елка" - тест'


def test_exact_quote_is_verified(index: DocumentIndex) -> None:
    check = index.check("Начальная (максимальная) цена контракта составляет", 1)
    assert check.verified
    assert check.method == "exact"
    assert check.page == 1


def test_quote_found_on_another_page_fixes_page_number(index: DocumentIndex) -> None:
    """Модель ошиблась страницей, но цитата настоящая — подтверждение сохраняем."""
    check = index.check("устанавливается штраф в размере 3 (трёх) процентов", claimed_page=1)
    assert check.verified
    assert check.page == 2


def test_quote_with_typos_survives(index: DocumentIndex) -> None:
    """След OCR: пара искажённых символов не должна обнулять подтверждение."""
    check = index.check(
        "За каждый факт неисполнения подрядчикoм обязательства устанавливается штрaф "
        "в размере 3 (трёх) процентов цены контракта",
        2,
    )
    assert check.verified
    assert check.method == "fuzzy"


def test_fabricated_amount_is_rejected(index: DocumentIndex) -> None:
    check = index.check("Цена контракта составляет 99 999 999,00 рублей", 1)
    assert not check.verified


def test_paraphrase_is_rejected(index: DocumentIndex) -> None:
    check = index.check("Подрядчик обязан заплатить неустойку за срыв сроков поставки", 2)
    assert not check.verified


def test_too_short_quote_is_rejected(index: DocumentIndex) -> None:
    assert not index.check("НДС", 1).verified


def test_quote_ignores_case_and_yo(index: DocumentIndex) -> None:
    check = index.check("В РАЗМЕРЕ 3 (ТРЕХ) ПРОЦЕНТОВ ЦЕНЫ КОНТРАКТА", 2)
    assert check.verified
