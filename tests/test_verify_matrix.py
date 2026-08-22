"""Матрица сверки цитат — главный контроль качества сервиса.

Каждый случай здесь появился из настоящей дыры: подмена суммы, подмена цифры
внутри суммы, подмена месяца прописью и дописанный к настоящей цитате хвост
раньше проходили проверку и получали verified=true. Таблица в SOLUTION.md
описывает ровно этот набор, поэтому она не может разъехаться с поведением
незамеченной.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.extraction.verify import DocumentIndex
from app.services.pdf.extractor import extract_document

# (описание, цитата, заявленная страница, должна ли подтвердиться)
CASES: list[tuple[str, str, int, bool]] = [
    ("точная цитата", "Начальная (максимальная) цена контракта составляет 12 480 350,00", 1, True),
    ("цитата с другой страницы", "штраф в размере 3 (трёх) процентов цены контракта", 1, True),
    (
        "потеряна скобочная расшифровка",
        "Начальная (максимальная) цена контракта составляет 12 480 350,00 рублей 00 копеек",
        1,
        True,
    ),
    (
        "потеряны скобки целиком",
        "Начальная максимальная цена контракта составляет 12 480 350,00",
        1,
        True,
    ),
    (
        "опечатки как после OCR",
        "Гарантийный срок на резульат выполненых работ составляет 60 (шестьдесят) месяцев",
        4,
        True,
    ),
    ("потеряны копейки", "Начальная (максимальная) цена контракта составляет 12 480 350", 1, True),
    (
        "подменена сумма",
        "Начальная (максимальная) цена контракта составляет 99 999 999,00",
        1,
        False,
    ),
    (
        "подменена одна цифра в сумме",
        "Начальная (максимальная) цена контракта составляет 12 480 950,00",
        1,
        False,
    ),
    (
        "подменён месяц прописью",
        "Работы выполняются с даты заключения контракта по 20 ноября 2026 года включительно",
        1,
        False,
    ),
    (
        "подменён процент",
        "устанавливается штраф в размере 30 (тридцати) процентов цены контракта",
        4,
        False,
    ),
    (
        "к настоящей цитате дописан хвост",
        "Начальная (максимальная) цена контракта составляет 12 480 350,00 рублей, "
        "авансирование составляет тридцать процентов",
        1,
        False,
    ),
    ("пересказ своими словами", "Подрядчик обязан завершить все работы до конца года", 1, False),
    (
        "подмена условия",
        "Аванс составляет 30 процентов цены контракта и перечисляется в течение трёх дней",
        1,
        False,
    ),
    (
        "цитата из другого документа",
        "Поставщик обязан обеспечить гарантийное обслуживание медицинского оборудования",
        1,
        False,
    ),
]


@pytest.fixture(scope="module")
def index(request: pytest.FixtureRequest) -> DocumentIndex:
    from tests.conftest import SAMPLES

    document = extract_document(
        (SAMPLES / "tender-44fz-remont-krovli.pdf").read_bytes(),
        Settings(ocr_enabled=False, _env_file=None),
    )
    return DocumentIndex([(page.number, page.text) for page in document.pages])


@pytest.mark.parametrize(
    ("description", "quote", "page", "expected"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_quote_verification(
    index: DocumentIndex, description: str, quote: str, page: int, expected: bool
) -> None:
    check = index.check(quote, page)
    assert check.verified is expected, (
        f"{description}: ожидалось verified={expected}, получено {check.verified} "
        f"(метод {check.method}, оценка {check.score:.2f})"
    )


def test_substitution_is_reported_as_mismatch(index: DocumentIndex) -> None:
    """Подмена отличается от «цитаты вообще нет» — это разные диагнозы."""
    substituted = index.check("Начальная (максимальная) цена контракта составляет 99 999 999,00", 1)
    absent = index.check("Подрядчик обязан завершить все работы до конца года", 1)

    assert substituted.method == "mismatch"
    assert substituted.is_mismatch
    assert absent.method == "not_found"
    assert not absent.is_mismatch


def test_quote_without_page_is_still_checked(index: DocumentIndex) -> None:
    """Модель может не указать страницу — искать цитату это не мешает."""
    check = index.check("Начальная (максимальная) цена контракта составляет 12 480 350,00", None)
    assert check.verified
    assert check.page == 1
