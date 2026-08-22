"""Сверка цитат-подтверждений с исходным текстом документа.

Это ключевой контроль качества всего сервиса. Модель обязана к каждому
извлечённому значению приложить дословную цитату; здесь цитата ищется в
тексте, который реально был извлечён из PDF. Если её там нет — значение
помечается как неподтверждённое, попадает в warnings и понижает confidence.

Сравнение идёт по нормализованному тексту: регистр, «ё», типы кавычек и
тире, неразрывные пробелы и переносы строк в документах пляшут, и требовать
побайтового совпадения означало бы браковать корректные цитаты.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

VerificationMethod = Literal["exact", "fuzzy", "not_found"]

_QUOTE_CHARS = str.maketrans(
    {
        "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'",
        "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-",
        " ": " ", " ": " ", " ": " ", " ": " ", "­": "",
        "ё": "е", "Ё": "Е",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")

# Ниже этой доли совпадения цитата считается не найденной. Порог подобран так,
# чтобы пережить мелкие расхождения OCR, но не пропустить пересказ.
FUZZY_THRESHOLD = 0.82
MIN_QUOTE_LENGTH = 8


def normalize(text: str) -> str:
    """Приводит текст к виду, в котором его можно сравнивать."""
    return _WHITESPACE_RE.sub(" ", text.translate(_QUOTE_CHARS).lower()).strip()


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    """Результат сверки одной цитаты."""

    verified: bool
    page: int | None
    method: VerificationMethod
    score: float = 0.0


class DocumentIndex:
    """Нормализованный текст документа, подготовленный для поиска цитат.

    Нормализация каждой страницы выполняется один раз на документ: цитат
    в разборе десятки, и перенормализовывать текст под каждую — впустую
    жечь процессор.
    """

    def __init__(self, pages: list[tuple[int, str]]) -> None:
        self._pages: dict[int, str] = {number: normalize(text) for number, text in pages}

    @property
    def page_numbers(self) -> list[int]:
        return sorted(self._pages)

    def check(self, quote: str, claimed_page: int | None = None) -> QuoteCheck:
        """Ищет цитату в документе.

        Сначала точное вхождение на заявленной странице, затем на остальных
        (модель регулярно ошибается со страницей на границе фрагментов — это
        не выдумка, и терять такое подтверждение неправильно), затем — нечёткое
        сравнение, которое спасает цитаты со страниц, прошедших через OCR.
        """
        needle = normalize(quote)
        if len(needle) < MIN_QUOTE_LENGTH:
            return QuoteCheck(verified=False, page=claimed_page, method="not_found")

        if claimed_page in self._pages and needle in self._pages[claimed_page]:
            return QuoteCheck(verified=True, page=claimed_page, method="exact", score=1.0)

        for number, text in self._pages.items():
            if needle in text:
                return QuoteCheck(verified=True, page=number, method="exact", score=1.0)

        best_page, best_score = None, 0.0
        for number, text in self._pages.items():
            score = _best_window_ratio(text, needle)
            if score > best_score:
                best_page, best_score = number, score

        if best_score >= FUZZY_THRESHOLD:
            return QuoteCheck(verified=True, page=best_page, method="fuzzy", score=best_score)
        return QuoteCheck(verified=False, page=claimed_page, method="not_found", score=best_score)


def _best_window_ratio(haystack: str, needle: str) -> float:
    """Какая доля цитаты действительно присутствует в тексте.

    Сначала самый длинный общий кусок задаёт место выравнивания, затем в окне
    вокруг него считается суммарная длина совпавших фрагментов цитаты. Метрика
    односторонняя (полнота цитаты, а не схожесть двух строк) и потому не
    штрафует за вставки в документе: скобочная расшифровка суммы посреди
    предложения — обычное дело. Фрагменты короче четырёх символов не считаются,
    иначе совпадения предлогов и пробелов надули бы оценку любому тексту.
    """
    if not needle or not haystack:
        return 0.0

    matcher = SequenceMatcher(None, haystack, needle, autojunk=False)
    anchor = matcher.find_longest_match(0, len(haystack), 0, len(needle))
    if anchor.size < max(MIN_QUOTE_LENGTH, len(needle) * 0.2):
        # Общего куска приличной длины нет — выравнивать не по чему.
        return anchor.size / len(needle)

    padding = max(40, len(needle) // 2)
    aligned = anchor.a - anchor.b
    window = haystack[max(0, aligned - padding) : min(len(haystack), aligned + len(needle) + padding)]

    matched = sum(
        block.size
        for block in SequenceMatcher(None, window, needle, autojunk=False).get_matching_blocks()
        if block.size >= 4
    )
    return min(1.0, matched / len(needle))
