"""Сверка цитат-подтверждений с исходным текстом документа.

Это ключевой контроль качества всего сервиса. Модель обязана к каждому
извлечённому значению приложить дословную цитату; здесь цитата ищется в
тексте, который реально был извлечён из PDF. Если её там нет — значение
помечается как неподтверждённое, попадает в warnings и понижает confidence.

Сравнение идёт по нормализованному тексту: регистр, «ё», типы кавычек и
тире, неразрывные пробелы и переносы строк в документах пляшут, и требовать
побайтового совпадения означало бы браковать корректные цитаты.

Нечёткое сравнение устроено в три слоя, и каждый закрывает свою дыру:

* доля цитаты, найденная в тексте, — отсеивает пересказ;
* сверка чисел — отсеивает подмену суммы или даты, при которой предложение
  скопировано дословно, а цифра заменена (текстовое сходство при этом
  остаётся высоким, и одной лишь оценки похожести не хватает);
* пословная сверка — отсеивает подмену слова, при которой цифры совпадают:
  «по 20 декабря» против «по 20 ноября». Слово считается найденным, если в
  тексте есть достаточно похожее, поэтому опечатки OCR она переживает.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

VerificationMethod = Literal["exact", "fuzzy", "mismatch", "not_found"]

# Записано кодами, а не самими символами: разные пробелы и тире визуально
# неотличимы, и в исходнике их слишком легко потерять при копировании.
_QUOTE_CHARS = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2009": " ",
        "\u2007": " ",
        "\u00ad": "",
        "ё": "е",
        "Ё": "Е",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
# Число вместе с разделителями разрядов и копейками: «12 480 350,00».
_NUMBER_RE = re.compile(r"\d[\d .,]*\d|\d")
_WORD_RE = re.compile(r"[\w./-]+")

# Ниже этой доли совпадения цитата считается не найденной. Порог работает не
# один: любое нечёткое совпадение дополнительно проходит сверку чисел и слов.
FUZZY_THRESHOLD = 0.75
MIN_QUOTE_LENGTH = 8
# Числа короче трёх цифр (проценты, номера пунктов) слишком часто совпадают
# случайно, чтобы что-то доказывать.
MIN_SIGNIFICANT_DIGITS = 3
# Слова короче этого — предлоги и союзы, их совпадение ничего не значит.
MIN_SIGNIFICANT_WORD = 4
# Насколько слово цитаты должно быть похоже на слово из текста, чтобы считаться
# тем же словом. Подобрано так, чтобы «резульат» сошлось с «результат»,
# а «ноября» с «декабря» — нет.
WORD_SIMILARITY = 0.75


def normalize(text: str) -> str:
    """Приводит текст к виду, в котором его можно сравнивать."""
    return _WHITESPACE_RE.sub(" ", text.translate(_QUOTE_CHARS).lower()).strip()


def digit_tokens(text: str) -> set[str]:
    """Числа текста, сведённые к одним цифрам: «12 480 350,00» → «1248035000»."""
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        digits = "".join(char for char in match.group(0) if char.isdigit())
        if len(digits) >= MIN_SIGNIFICANT_DIGITS:
            tokens.add(digits)
    return tokens


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    """Результат сверки одной цитаты."""

    verified: bool
    page: int | None
    method: VerificationMethod
    score: float = 0.0

    @property
    def is_mismatch(self) -> bool:
        """Текст похож, но числа или слова не сошлись — признак подмены."""
        return self.method == "mismatch"


class DocumentIndex:
    """Нормализованный текст документа, подготовленный для поиска цитат.

    Нормализация каждой страницы выполняется один раз на документ: цитат
    в разборе десятки, и перенормализовывать текст под каждую — впустую
    жечь процессор.
    """

    def __init__(self, pages: list[tuple[int, str]]) -> None:
        self._pages: dict[int, str] = {number: normalize(text) for number, text in pages}
        self._numbers: dict[int, set[str]] = {
            number: digit_tokens(text) for number, text in self._pages.items()
        }

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

        best_page, best_score, best_window = None, 0.0, ""
        for number, text in self._pages.items():
            score, window = _align(text, needle)
            if score > best_score:
                best_page, best_score, best_window = number, score, window

        if best_score < FUZZY_THRESHOLD or best_page is None:
            return QuoteCheck(
                verified=False, page=claimed_page, method="not_found", score=best_score
            )

        # Текст похож — остаётся убедиться, что в нём ничего не подменили.
        numbers_ok = _numbers_supported(digit_tokens(needle), self._numbers[best_page])
        words_ok = _words_supported(needle, best_window)
        if numbers_ok and words_ok:
            return QuoteCheck(verified=True, page=best_page, method="fuzzy", score=best_score)
        return QuoteCheck(verified=False, page=best_page, method="mismatch", score=best_score)


def _align(haystack: str, needle: str) -> tuple[float, str]:
    """Находит участок текста, наиболее похожий на цитату.

    Возвращает долю цитаты, найденную в этом участке, и сам участок. Метрика
    односторонняя — это полнота цитаты, а не схожесть двух строк. Обычная
    схожесть штрафовала бы за вставки в документе, а скобочная расшифровка
    суммы посреди предложения там норма. Фрагменты короче четырёх символов
    не считаются, иначе совпадения предлогов надули бы оценку любому тексту.
    """
    if not needle or not haystack:
        return 0.0, ""

    matcher = SequenceMatcher(None, haystack, needle, autojunk=False)
    anchor = matcher.find_longest_match(0, len(haystack), 0, len(needle))
    if anchor.size < max(MIN_QUOTE_LENGTH, len(needle) * 0.2):
        # Общего куска приличной длины нет — выравнивать не по чему.
        return anchor.size / len(needle), ""

    # Окно заметно шире цитаты: документ вставляет внутрь предложения то, чего
    # в цитате нет, и при тесном окне законный хвост цитаты оказался бы за
    # его границей и сошёл бы за выдумку.
    padding = max(80, len(needle))
    aligned = anchor.a - anchor.b
    window = haystack[
        max(0, aligned - padding) : min(len(haystack), aligned + len(needle) + padding)
    ]

    matched = sum(
        block.size
        for block in SequenceMatcher(None, window, needle, autojunk=False).get_matching_blocks()
        if block.size >= 4
    )
    return min(1.0, matched / len(needle)), window


def _numbers_supported(quote_numbers: set[str], page_numbers: set[str]) -> bool:
    """Все ли значимые числа цитаты действительно есть на странице.

    Сравнение с допуском: цитата могла потерять копейки («12 480 350» вместо
    «12 480 350,00»), поэтому число засчитывается, если оно является началом
    числа со страницы или наоборот.
    """
    return all(
        any(number in candidate or candidate in number for candidate in page_numbers)
        for number in quote_numbers
    )


def _is_significant(word: str) -> bool:
    """Стоит ли требовать для слова соответствия в тексте.

    Короткие числа («12» из разрядов суммы, «60» из срока гарантии) отдельно
    ничего не доказывают: они рассыпаны по любому документу, а их сверка —
    работа _numbers_supported, которая смотрит на число целиком.
    """
    if any(char.isdigit() for char in word):
        return len(word) >= MIN_SIGNIFICANT_DIGITS
    return len(word) >= MIN_SIGNIFICANT_WORD


def _words_supported(needle: str, window: str) -> bool:
    """Каждому значимому слову цитаты нашлось соответствие в тексте.

    Именно эта проверка ловит подмену слова при совпадающих цифрах. Сравнение
    нестрогое, поэтому искажённое распознаванием слово всё равно находит свой
    оригинал, а подставленное — нет.
    """
    if not window:
        return False

    window_words = set(_WORD_RE.findall(window))
    if not window_words:
        return False

    for word in _WORD_RE.findall(needle):
        if not _is_significant(word):
            continue
        if word in window_words:
            continue
        if any(
            SequenceMatcher(None, word, candidate, autojunk=False).ratio() >= WORD_SIMILARITY
            for candidate in window_words
        ):
            continue
        return False
    return True
