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


def test_amounts_with_dot_separators() -> None:
    """«12.480.350,00» — редкое, но встречающееся написание разрядов.

    Раньше регулярное выражение цеплялось за хвост числа и возвращало 350.00,
    промахиваясь на пять порядков.
    """
    hits = heuristics.find_money([(1, "цена контракта составляет 12.480.350,00 рублей")])
    assert [hit.amount for hit in hits] == [Decimal("12480350.00")]


def test_penalty_marker_requires_word_boundary() -> None:
    """«степень» и «компенсационный» содержат «пен», но санкциями не являются."""
    text = (
        "Гарантийный срок на конструкции устанавливается в зависимости от степени износа "
        "и подтверждается взносом в компенсационный фонд саморегулируемой организации."
    )
    assert heuristics.find_penalties([(1, text)]) == []


def test_fine_is_not_classified_as_penya() -> None:
    assert (
        heuristics.classify_penalty(
            "подрядчик уплачивает штраф, размер которого зависит от степени тяжести нарушения"
        )
        == "штраф"
    )


def test_penya_wins_when_formula_present() -> None:
    assert (
        heuristics.classify_penalty(
            "начисляется в размере одной трёхсотой ключевой ставки за каждый день просрочки"
        )
        == "пеня"
    )


def test_nearest_date_wins_over_format() -> None:
    """Раньше дата цифрами побеждала более близкую к якорю дату словами."""
    text = (
        "Дата и время окончания подачи заявок: 15 сентября 2026 г., 09:00. "
        "Дата подачи ценовых предложений: 18.09.2026."
    )
    findings = heuristics.analyze([(1, text)])
    assert findings.application_deadline is not None
    assert findings.application_deadline.value == date(2026, 9, 15)


def test_law_144_is_not_law_44() -> None:
    assert heuristics.detect_law("закупка по Федеральному закону № 144-ФЗ") != "44-ФЗ"


def test_procurement_number_for_223fz() -> None:
    """По 223-ФЗ номер закупки одиннадцатизначный, а не девятнадцатизначный."""
    assert (
        heuristics.detect_procurement_number("Извещение № 32211456789 от 01.09.2026")
        == "32211456789"
    )


def test_security_anchor_knows_about_dogovor() -> None:
    """По 223-ФЗ заключают договор, а не контракт."""
    text = "Размер обеспечения исполнения договора: 199 000,00 рублей."
    findings = heuristics.analyze([(1, text)])
    assert findings.contract_security is not None
    assert findings.contract_security.amount == Decimal("199000.00")


def test_anchor_survives_line_break_inside_label() -> None:
    """В таблице ярлык переносится на другую строку и подстрокой не находится."""
    text = "Размер обеспечения исполнения\nконтракта 624 017,50 рубля"
    findings = heuristics.analyze([(1, text)])
    assert findings.contract_security is not None
    assert findings.contract_security.amount == Decimal("624017.50")


def test_several_prices_near_anchor_are_all_collected() -> None:
    """Документ может называть цену дважды и по-разному — это надо заметить."""
    pages = [
        (1, "Начальная (максимальная) цена контракта составляет 5 460 000,00 рублей."),
        (2, "Начальная (максимальная) цена контракта составляет 5 600 000,00 рублей."),
    ]
    findings = heuristics.analyze(pages)
    assert findings.price_candidates == {Decimal("5460000.00"), Decimal("5600000.00")}
