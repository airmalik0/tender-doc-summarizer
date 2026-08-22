"""Offline-провайдер: извлечение без обращения к LLM.

Это не заглушка и не моки. Провайдер получает ровно тот же вход, что и
модель, и обязан вернуть ровно тот же контракт — только внутри у него не
нейросеть, а детерминированные правила из app/services/extraction/heuristics.

Зачем он нужен:

* сервис поднимается и делает осмысленную работу без единого API-ключа —
  проверяющему достаточно `docker compose up`;
* тесты пайплайна гоняются без сети и без денег;
* когда провайдер лежит или ключ протух, сервис отвечает деградировавшим
  результатом с честным warning, а не пятисоткой.

Качество ниже, чем у модели, и это указывается в warnings ответа. Зато
каждая цитата здесь заведомо настоящая: она вырезана из самого документа,
поэтому сверка подтверждает её всегда.
"""

from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel

from app.services.extraction import heuristics
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage

PAGE_MARKER_RE = re.compile(r"\[СТРАНИЦА\s+(\d+)\]")

_SUBJECT_ANCHORS = ("объект закупки", "предмет закупки", "наименование объекта закупки")
_CUSTOMER_ANCHORS = ("наименование заказчика", "заказчик:", "сведения о заказчике")

_REQUIREMENT_MARKERS = (
    "должен", "должны", "обязан", "требовани", "наличие у участника", "участник закупки",
    "соответствие", "не менее", "предоставля", "гарантийный срок", "отсутствие",
)
_REQUIREMENT_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("лицензии и допуски", ("саморегулируем", "сро", "лицензи", "разрешени", "удостоверен", "допуск", "реестр специалистов")),
    ("опыт", ("опыт", "исполненн", "за последние", "аналогичн", "ранее заключ")),
    ("квалификация персонала", ("специалист", "в штате", "квалифика", "образовани", "персонал")),
    ("материально-техническая база", ("оборудован", "техник", "материальн", "ресурс", "сервисн")),
    ("финансовые гарантии", ("обеспечен", "независим", "банковск", "залог")),
    ("требования к товару или работам", ("товар", "гост", "срок годности", "новым", "поставля", "гарантийный срок")),
)
_OPTIONAL_MARKERS = ("вправе", "может быть", "по желанию", "рекомендуется")

_EXECUTOR_WORDS = ("подрядчик", "поставщик", "исполнител", "участник")
_CUSTOMER_WORDS = ("заказчик",)


class OfflineProvider(LLMProvider):
    """Правила вместо модели. Тот же контракт, пониженное качество."""

    name = "offline"
    supports_synthesis = False

    @property
    def model(self) -> str | None:
        return None

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        max_output_tokens: int = 16_000,
    ) -> LLMResult:
        started = time.perf_counter()
        pages = split_pages(user)
        data = build_extraction(pages)
        return LLMResult(
            data=data,
            usage=LLMUsage(),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def split_pages(prompt_text: str) -> list[tuple[int, str]]:
    """Восстанавливает разбиение по страницам из маркеров в промпте."""
    matches = list(PAGE_MARKER_RE.finditer(prompt_text))
    if not matches:
        return [(1, prompt_text)]

    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt_text)
        pages.append((int(match.group(1)), prompt_text[match.end() : end].strip()))
    return pages


def build_extraction(pages: list[tuple[int, str]]) -> dict[str, Any]:
    """Собирает ChunkExtraction по результатам детерминированного разбора."""
    findings = heuristics.analyze(pages)

    return {
        "subject": _anchored_value(pages, _SUBJECT_ANCHORS),
        "customer": _anchored_value(pages, _CUSTOMER_ANCHORS),
        "procurement_number": findings.procurement_number,
        "law": findings.law,
        "price": _money(findings.price, vat_hint=_vat_hint(pages)),
        "contract_security": _money(findings.contract_security, vat_hint="не указано"),
        "application_deadline": _date(findings.application_deadline),
        "contract_start": None,
        "contract_end": None,
        "duration_text": _duration_text(pages),
        "stages": [],
        "requirements": _requirements(pages),
        "penalties": _penalties(findings.penalties),
    }


def _money(hit: heuristics.MoneyHit | None, vat_hint: str) -> dict[str, Any] | None:
    if hit is None:
        return None
    return {
        "amount": float(hit.amount),
        "currency": "RUB",
        "vat": vat_hint,
        "raw_text": hit.raw,
        "evidence": {"page": hit.page, "quote": hit.context[:300]},
    }


def _date(hit: heuristics.DateHit | None) -> dict[str, Any] | None:
    if hit is None:
        return None
    return {
        "iso_date": hit.value.isoformat(),
        "raw_text": hit.raw,
        "evidence": {"page": hit.page, "quote": hit.raw},
    }


def _vat_hint(pages: list[tuple[int, str]]) -> str:
    text = " ".join(page_text for _, page_text in pages).lower()
    if "не облагается" in text or "без ндс" in text:
        return "не включён"
    if "в том числе ндс" in text or "включая ндс" in text:
        return "включён"
    return "не указано"


def _anchored_value(pages: list[tuple[int, str]], anchors: tuple[str, ...]) -> str | None:
    """Значение, стоящее сразу после ярлыка в таблице реквизитов."""
    for _, text in pages:
        lowered = text.lower()
        for anchor in anchors:
            position = lowered.find(anchor)
            if position == -1:
                continue
            tail = text[position + len(anchor) : position + len(anchor) + 220]
            value = " ".join(tail.strip(" :\n\t").split())
            if len(value) > 12:
                return value[:200]
    return None


_DURATION_ANCHORS = ("срок выполнения работ", "срок поставки товара", "срок поставки", "срок оказания услуг")


def _duration_text(pages: list[tuple[int, str]]) -> str | None:
    """Формулировка срока исполнения, начиная ровно с якоря.

    Резать по границе предложения нельзя: в таблицах реквизитов строки часто
    идут без точек, и одно «предложение» склеивает несколько разных сроков.
    """
    for _, text in pages:
        lowered = text.lower()
        for anchor in _DURATION_ANCHORS:
            position = lowered.find(anchor)
            if position == -1:
                continue
            fragment = " ".join(text[position : position + 300].split())
            head, separator, _ = fragment.partition(". ")
            return (head + separator).strip() if separator else fragment
    return None


def _requirements(pages: list[tuple[int, str]]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()

    for page_number, text in pages:
        for sentence in _sentences(text):
            lowered = sentence.lower()
            if not (40 <= len(sentence) <= 420):
                continue
            if not any(marker in lowered for marker in _REQUIREMENT_MARKERS):
                continue
            key = lowered[:70]
            if key in seen:
                continue
            seen.add(key)
            requirements.append(
                {
                    "text": sentence[:250],
                    "category": _categorize(lowered),
                    "mandatory": not any(word in lowered for word in _OPTIONAL_MARKERS),
                    "evidence": {"page": page_number, "quote": sentence[:300]},
                }
            )
    return requirements[:15]


def _categorize(lowered: str) -> str:
    for category, markers in _REQUIREMENT_CATEGORIES:
        if any(marker in lowered for marker in markers):
            return category
    return "прочее"


def _penalties(hits: list[heuristics.PenaltyHit]) -> list[dict[str, Any]]:
    penalties: list[dict[str, Any]] = []
    seen: set[str] = set()

    for hit in hits:
        key = hit.sentence.lower()[:70]
        if key in seen:
            continue
        seen.add(key)

        lowered = hit.sentence.lower()
        penalties.append(
            {
                "kind": hit.kind,
                "party": _party(lowered),
                "trigger": hit.sentence[:250],
                "calculation": _calculation(hit.sentence),
                "amount_text": _amount_text(hit.sentence),
                "evidence": {"page": hit.page, "quote": hit.sentence[:300]},
            }
        )
    return penalties[:12]


def _party(lowered: str) -> str:
    has_executor = any(word in lowered for word in _EXECUTOR_WORDS)
    has_customer = any(word in lowered for word in _CUSTOMER_WORDS)
    if has_executor and not has_customer:
        return "исполнитель"
    if has_customer and not has_executor:
        return "заказчик"
    if has_executor and has_customer:
        return "обе стороны"
    return "не указано"


def _calculation(sentence: str) -> str | None:
    match = heuristics.PENALTY_FORMULA_RE.search(sentence)
    if not match:
        return None
    start = max(0, match.start() - 60)
    return " ".join(sentence[start : match.end() + 140].split())


def _amount_text(sentence: str) -> str | None:
    money = heuristics.MONEY_RE.search(sentence)
    if money:
        return " ".join(money.group(0).split())
    percent = heuristics.PERCENT_RE.search(sentence)
    if percent:
        return " ".join(percent.group(0).split())
    return None


def _sentences(text: str) -> list[str]:
    return [
        " ".join(sentence.split())
        for sentence in heuristics._SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
        if sentence.strip()
    ]
