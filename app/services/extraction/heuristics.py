"""Детерминированный извлекатель на регулярных выражениях.

Зачем он нужен, если есть LLM. Во-первых, это независимая точка зрения:
если модель назвала ценой контракта сумму, которой в документе нет, парсер
об этом промолчит — и расхождение попадёт в warnings, а достоверность
снизится. Во-вторых, он позволяет сервису работать вообще без ключа: тот же
код обслуживает offline-провайдера.

Регулярные выражения написаны под язык российской закупочной документации:
разряды, отделённые неразрывными пробелами, прописная расшифровка в скобках,
даты словами, формулы пеней из ПП РФ от 30.08.2017 № 1042.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

# Разделителем разрядов в документах бывает и обычный пробел, и неразрывный,
# и узкий неразрывный — все три встречаются в выгрузках из Word.
_THIN_SPACES = "    "
# Три написания подряд: через точки («12.480.350,00» — только вместе с
# копейками, иначе «1.234» неотличимо от дробного числа), через пробелы любой
# ширины и просто числом.
_NUMBER = (
    rf"\d{{1,3}}(?:\.\d{{3}})+,\d{{1,2}}"
    rf"|\d{{1,3}}(?:[ {_THIN_SPACES}]\d{{3}})+(?:[.,]\d{{1,2}})?"
    rf"|\d+(?:[.,]\d{{1,2}})?"
)

# «12 480 350,00 (двенадцать миллионов ...) рублей 00 копеек» — скобочная
# расшифровка стоит между числом и словом «рублей», поэтому она пропускается.
MONEY_RE = re.compile(
    rf"(?P<amount>{_NUMBER})\s*(?:\([^)]{{0,220}}\)\s*)?(?P<unit>руб(?:\.|лей|ля|ль)?|₽)",
    re.IGNORECASE,
)

PERCENT_RE = re.compile(rf"(?P<value>{_NUMBER})\s*(?:\([^)]{{0,80}}\)\s*)?(?:%|процент\w*)", re.I)

DATE_DIGITS_RE = re.compile(r"\b(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})\b")

_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
DATE_WORDS_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+(?P<month>" + "|".join(_MONTHS) + r")\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

# Формулировки, рядом с которыми в документе стоит НМЦК.
PRICE_ANCHORS = (
    "начальная (максимальная) цена контракта",
    "начальная (максимальная) цена договора",
    "начальная (максимальная) цена",
    "цена договора составляет",
    "начальная(максимальная) цена контракта",
    "начальная максимальная цена контракта",
    "начальная (максимальная) цена",
    "нмцк",
    "цена контракта составляет",
    "цена контракта:",
)
SECURITY_ANCHORS = (
    "размер обеспечения исполнения контракта",
    "размер обеспечения исполнения договора",
    "обеспечение исполнения контракта",
    "обеспечение исполнения договора",
    "обеспечения исполнения договора",
    "обеспечения исполнения контракта",
    "размер обеспечения исполнения",
)
DEADLINE_ANCHORS = (
    "окончания подачи заявок",
    "окончание подачи заявок",
    "окончания срока подачи заявок",
    "дата и время окончания подачи",
)

LAW_44_RE = re.compile(
    r"(?<!\d)44[\s-]*фз|№\s*(?<!\d)44-фз|федеральн\w+ закон\w*[^.]{0,40}(?<!\d)44", re.IGNORECASE
)
LAW_223_RE = re.compile(r"223[\s-]*фз|№\s*223-фз", re.IGNORECASE)
# 19 цифр — извещение по 44-ФЗ, 11 цифр — по 223-ФЗ.
PROCUREMENT_NUMBER_RE = re.compile(r"\b(?:\d{19}|\d{11})\b")

# Границы слова обязательны: без них «пен» находится в «степени», и предложение
# про гарантийный срок превращается в меру ответственности.
PENALTY_PENYA_RE = re.compile(r"\bпен[иеяю]", re.IGNORECASE)
PENALTY_FINE_RE = re.compile(r"\bштраф", re.IGNORECASE)
PENALTY_FORFEIT_RE = re.compile(r"\bнеустойк", re.IGNORECASE)
PENALTY_FORMULA_RE = re.compile(
    r"одной\s+трёхсотой|одной\s+трехсотой|1/300|ключевой\s+ставк", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[А-ЯЁ«\d])")


def _flatten(text: str) -> str:
    """Текст без переносов строк, символ в символ.

    Ярлык в таблице сплошь и рядом разрывается переносом («Размер обеспечения
    исполнения\nконтракта»), и поиск якоря подстрокой его не находит. Замена
    один в один сохраняет позиции символов, поэтому найденное смещение можно
    без пересчёта использовать в исходном тексте.
    """
    return text.replace("\n", " ").lower()


@dataclass(frozen=True, slots=True)
class MoneyHit:
    """Денежная сумма, найденная в тексте."""

    amount: Decimal
    raw: str
    page: int
    context: str


@dataclass(frozen=True, slots=True)
class DateHit:
    """Дата, найденная в тексте."""

    value: date
    raw: str
    page: int


@dataclass(frozen=True, slots=True)
class PenaltyHit:
    """Предложение, описывающее меру ответственности."""

    kind: str
    sentence: str
    page: int


@dataclass(slots=True)
class RuleFindings:
    """Всё, что удалось получить без участия модели."""

    money: list[MoneyHit]
    dates: list[DateHit]
    penalties: list[PenaltyHit]
    price: MoneyHit | None
    price_candidates: set[Decimal]
    contract_security: MoneyHit | None
    application_deadline: DateHit | None
    law: str
    procurement_number: str | None

    @property
    def amounts(self) -> set[Decimal]:
        return {hit.amount for hit in self.money}


def parse_amount(raw: str) -> Decimal | None:
    """«12 480 350,00» → Decimal('12480350.00')."""
    cleaned = raw.strip()
    for space in (" ", *_THIN_SPACES):
        cleaned = cleaned.replace(space, "")
    cleaned = cleaned.replace(",", ".")
    if cleaned.count(".") > 1:  # «1.234.567» — точки как разделители разрядов
        head, _, tail = cleaned.rpartition(".")
        cleaned = head.replace(".", "") + "." + tail
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def parse_iso_date(raw: str | None) -> date | None:
    """Разбирает дату из ответа модели, прощая распространённые отклонения."""
    if not raw:
        return None
    text = raw.strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    match = DATE_DIGITS_RE.search(raw)
    if match:
        return _safe_date(match.group("year"), match.group("month"), match.group("day"))
    match = DATE_WORDS_RE.search(raw)
    if match:
        return _safe_date(
            match.group("year"), _MONTHS[match.group("month").lower()], match.group("day")
        )
    return None


def _safe_date(year: str | int, month: str | int, day: str | int) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def find_money(pages: list[tuple[int, str]]) -> list[MoneyHit]:
    """Все денежные суммы документа с окружающим контекстом."""
    hits: list[MoneyHit] = []
    for page_number, text in pages:
        for match in MONEY_RE.finditer(text):
            amount = parse_amount(match.group("amount"))
            if amount is None:
                continue
            start = max(0, match.start() - 160)
            hits.append(
                MoneyHit(
                    amount=amount,
                    raw=match.group(0).strip(),
                    page=page_number,
                    context=text[start : match.end() + 40].replace("\n", " "),
                )
            )
    return hits


def find_dates(pages: list[tuple[int, str]]) -> list[DateHit]:
    """Все даты документа в обоих распространённых написаниях."""
    hits: list[DateHit] = []
    for page_number, text in pages:
        for match in DATE_DIGITS_RE.finditer(text):
            value = _safe_date(match.group("year"), match.group("month"), match.group("day"))
            if value:
                hits.append(DateHit(value=value, raw=match.group(0), page=page_number))
        for match in DATE_WORDS_RE.finditer(text):
            value = _safe_date(
                match.group("year"), _MONTHS[match.group("month").lower()], match.group("day")
            )
            if value:
                hits.append(DateHit(value=value, raw=match.group(0), page=page_number))
    return hits


def find_penalties(pages: list[tuple[int, str]]) -> list[PenaltyHit]:
    """Предложения, в которых описаны пени и штрафы."""
    hits: list[PenaltyHit] = []
    for page_number, text in pages:
        for sentence in _SENTENCE_SPLIT_RE.split(text.replace("\n", " ")):
            lowered = sentence.lower()
            if not (
                PENALTY_PENYA_RE.search(lowered)
                or PENALTY_FINE_RE.search(lowered)
                or PENALTY_FORFEIT_RE.search(lowered)
            ):
                continue
            if len(sentence) < 40:
                continue
            kind = classify_penalty(lowered)
            hits.append(
                PenaltyHit(kind=kind, sentence=" ".join(sentence.split()), page=page_number)
            )
    return hits


def classify_penalty(text: str) -> str:
    """Пеня или штраф.

    Штраф проверяется первым: предложение про штраф нередко упоминает и пени
    («помимо пеней, начисляется штраф…»), а вот формула одной трёхсотой
    ключевой ставки встречается только у пени.
    """
    lowered = text.lower()
    has_penya = bool(PENALTY_PENYA_RE.search(lowered))
    if PENALTY_FINE_RE.search(lowered) and not has_penya:
        return "штраф"
    if has_penya or PENALTY_FORMULA_RE.search(lowered):
        return "пеня"
    return "штраф"


def _nearest_money_after(
    text: str, position: int, page_number: int, window: int = 400
) -> tuple[MoneyHit, int] | None:
    """Первая денежная сумма правее указанной позиции и конец её вхождения."""
    match = MONEY_RE.search(text, position, position + window)
    if not match:
        return None
    amount = parse_amount(match.group("amount"))
    if amount is None:
        return None
    hit = MoneyHit(
        amount=amount,
        raw=" ".join(match.group(0).split()),
        page=page_number,
        context=" ".join(text[max(0, position - 40) : match.end() + 40].split()),
    )
    return hit, match.end()


def _find_anchored_money(
    pages: list[tuple[int, str]],
    anchors: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> MoneyHit | None:
    """Первая сумма после якорной формулировки.

    forbidden отсеивает ложные срабатывания на заголовках. Раздел «Обеспечение
    заявки и обеспечение исполнения контракта» содержит якорь обеспечения
    исполнения, но ближайшая к нему сумма — это обеспечение заявки. Если между
    якорем и числом встретилось запрещённое слово, кандидат отбрасывается
    и поиск продолжается со следующего вхождения якоря.
    """
    for anchor in anchors:
        for page_number, text in pages:
            lowered = _flatten(text)
            position = lowered.find(anchor)
            while position != -1:
                found = _nearest_money_after(text, position, page_number)
                if found is not None:
                    hit, money_end = found
                    span = lowered[position:money_end]
                    if not any(word in span for word in forbidden):
                        return hit
                position = lowered.find(anchor, position + 1)
    return None


def _find_anchored_date(pages: list[tuple[int, str]], anchors: tuple[str, ...]) -> DateHit | None:
    """Ближайшая к якорю дата, в каком бы виде она ни была записана.

    Сравниваются позиции, а не форматы: раньше сначала искалось написание
    цифрами по всему окну, и дата соседнего пункта, стоящая дальше по тексту,
    побеждала дату словами, стоящую вплотную к якорю.
    """
    for page_number, text in pages:
        lowered = _flatten(text)
        for anchor in anchors:
            position = lowered.find(anchor)
            if position == -1:
                continue

            window = text[position : position + 300]
            candidates: list[tuple[int, DateHit]] = []

            for match in DATE_DIGITS_RE.finditer(window):
                value = _safe_date(match.group("year"), match.group("month"), match.group("day"))
                if value:
                    candidates.append(
                        (match.start(), DateHit(value=value, raw=match.group(0), page=page_number))
                    )
            for match in DATE_WORDS_RE.finditer(window):
                value = _safe_date(
                    match.group("year"), _MONTHS[match.group("month").lower()], match.group("day")
                )
                if value:
                    candidates.append(
                        (match.start(), DateHit(value=value, raw=match.group(0), page=page_number))
                    )

            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
    return None


def collect_anchored_amounts(
    pages: list[tuple[int, str]], anchors: tuple[str, ...], forbidden: tuple[str, ...] = ()
) -> set[Decimal]:
    """Все различные суммы, стоящие у якорной формулировки.

    Нужно, чтобы заметить документ, в котором цена контракта названа дважды
    и по-разному: извещение говорит одно, проект контракта — другое. Одна
    выбранная сумма такое расхождение скрывает.
    """
    found: set[Decimal] = set()
    for anchor in anchors:
        for page_number, text in pages:
            lowered = _flatten(text)
            position = lowered.find(anchor)
            while position != -1:
                candidate = _nearest_money_after(text, position, page_number)
                if candidate is not None:
                    hit, money_end = candidate
                    if not any(word in lowered[position:money_end] for word in forbidden):
                        found.add(hit.amount)
                position = lowered.find(anchor, position + 1)
    return found


def detect_law(text: str) -> str:
    """Определяет закон, по которому проводится закупка."""
    has_44 = bool(LAW_44_RE.search(text))
    has_223 = bool(LAW_223_RE.search(text))
    if has_44 and not has_223:
        return "44-ФЗ"
    if has_223 and not has_44:
        return "223-ФЗ"
    if has_44 and has_223:
        # Оба упомянуты — обычно 223-ФЗ цитирует 44-ФЗ или наоборот.
        # Считаем основным тот, что встречается чаще.
        return (
            "44-ФЗ" if len(LAW_44_RE.findall(text)) >= len(LAW_223_RE.findall(text)) else "223-ФЗ"
        )
    return "не определено"


def detect_procurement_number(text: str) -> str | None:
    """Номер извещения в ЕИС — 19 цифр."""
    match = PROCUREMENT_NUMBER_RE.search(text)
    return match.group(0) if match else None


def analyze(pages: list[tuple[int, str]]) -> RuleFindings:
    """Полный детерминированный разбор документа."""
    full_text = "\n".join(text for _, text in pages)
    return RuleFindings(
        money=find_money(pages),
        dates=find_dates(pages),
        penalties=find_penalties(pages),
        price=_find_anchored_money(pages, PRICE_ANCHORS, forbidden=("обеспечен",)),
        price_candidates=collect_anchored_amounts(pages, PRICE_ANCHORS, forbidden=("обеспечен",)),
        contract_security=_find_anchored_money(
            pages, SECURITY_ANCHORS, forbidden=("заявк", "гарантийн")
        ),
        application_deadline=_find_anchored_date(pages, DEADLINE_ANCHORS),
        law=detect_law(full_text),
        procurement_number=detect_procurement_number(full_text),
    )
