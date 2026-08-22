"""Тесты детерминированного парсера.

Он — вторая опора всего сервиса: именно он подтверждает или не
подтверждает суммы, названные моделью. Ошибка здесь тихо портит
достоверность каждого разбора.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.services.extraction import heuristics


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12 480 350,00", Decimal("12480350.00")),
        ("12 480 350,00", Decimal("12480350.00")),  # неразрывные пробелы
        ("487620", Decimal("487620")),
        ("1 000,5", Decimal("1000.5")),
        ("2.850.000,00", Decimal("2850000.00")),
    ],
)
def test_parse_amount(raw: str, expected: Decimal) -> None:
    assert heuristics.parse_amount(raw) == expected


def test_parse_amount_returns_none_for_garbage() -> None:
    assert heuristics.parse_amount("не число") is None


def test_find_money_skips_prose_in_brackets() -> None:
    """Прописная расшифровка стоит между числом и словом «рублей»."""
    text = "цена контракта составляет 12 480 350,00 (двенадцать миллионов) рублей 00 копеек"
    hits = heuristics.find_money([(1, text)])
    assert [hit.amount for hit in hits] == [Decimal("12480350.00")]


def test_price_anchor_finds_nmck(roof_pdf: bytes, settings) -> None:
    from app.services.pdf.extractor import extract_document

    document = extract_document(roof_pdf, settings)
    findings = heuristics.analyze([(page.number, page.text) for page in document.pages])

    assert findings.price is not None
    assert findings.price.amount == Decimal("12480350.00")


def test_security_anchor_does_not_grab_bid_security(roof_pdf: bytes, settings) -> None:
    """Заголовок раздела содержит нужный якорь, но ближайшая сумма — чужая.

    «Обеспечение заявки и обеспечение исполнения контракта» — заголовок,
    сразу под которым идёт строка с обеспечением ЗАЯВКИ (124 803,50).
    Без отсечения по запрещённым словам парсер брал именно её.
    """
    from app.services.pdf.extractor import extract_document

    document = extract_document(roof_pdf, settings)
    findings = heuristics.analyze([(page.number, page.text) for page in document.pages])

    assert findings.contract_security is not None
    assert findings.contract_security.amount == Decimal("624017.50")
    assert findings.contract_security.amount != Decimal("124803.50")


def test_dates_in_both_notations() -> None:
    hits = heuristics.find_dates([(1, "подписан 05.04.2013, срок до 20 декабря 2026 года")])
    values = {hit.value for hit in hits}
    assert date(2013, 4, 5) in values
    assert date(2026, 12, 20) in values


def test_invalid_date_is_ignored() -> None:
    assert heuristics.find_dates([(1, "дата 32.13.2026")]) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("закупка по Федеральному закону № 44-ФЗ", "44-ФЗ"),
        ("положение о закупке по 223-ФЗ", "223-ФЗ"),
        ("никаких законов здесь нет", "не определено"),
    ],
)
def test_detect_law(text: str, expected: str) -> None:
    assert heuristics.detect_law(text) == expected


def test_procurement_number_is_nineteen_digits() -> None:
    text = "Извещение № 0118300012026000047 от 24.08.2026, ИНН 2310054871"
    assert heuristics.detect_procurement_number(text) == "0118300012026000047"


def test_penalty_sentences_are_classified() -> None:
    text = (
        "Пеня начисляется за каждый день просрочки в размере одной трёхсотой ключевой ставки "
        "Центрального банка Российской Федерации от цены контракта. "
        "За каждый факт неисполнения обязательства устанавливается штраф в размере "
        "374 410,50 рубля, что составляет три процента цены контракта."
    )
    hits = heuristics.find_penalties([(3, text)])
    assert {hit.kind for hit in hits} == {"пеня", "штраф"}
    assert all(hit.page == 3 for hit in hits)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-09-15", date(2026, 9, 15)),
        ("15.09.2026", date(2026, 9, 15)),
        ("15 сентября 2026 года", date(2026, 9, 15)),
        ("", None),
        ("в течение 45 дней", None),
    ],
)
def test_parse_iso_date(raw: str, expected: date | None) -> None:
    assert heuristics.parse_iso_date(raw) == expected
