"""Слияние результатов по фрагментам (шаг reduce).

Каждый фрагмент разбирался независимо, поэтому один и тот же факт
приходит по нескольку раз — иногда в разных формулировках, иногда с
разными значениями. Задача этого модуля — свести их в один набор фактов
и, что важнее, не спрятать расхождения: если два фрагмента называют разную
цену контракта, это не шум, а повод посмотреть документ глазами.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TypeVar

from app.models.extraction import (
    ChunkExtraction,
    RawDate,
    RawMoney,
    RawPenalty,
    RawRequirement,
    RawStage,
)
from app.services.extraction.verify import normalize

# Выше этого порога два требования (или две санкции) считаются одним и тем же.
SIMILARITY_THRESHOLD = 0.86

T = TypeVar("T")


@dataclass(slots=True)
class MergedFacts:
    """Факты по всему документу, сведённые из фрагментов."""

    subject: str | None = None
    customer: str | None = None
    procurement_number: str | None = None
    law: str | None = None
    price: RawMoney | None = None
    contract_security: RawMoney | None = None
    application_deadline: RawDate | None = None
    contract_start: RawDate | None = None
    contract_end: RawDate | None = None
    duration_text: str | None = None
    stages: list[RawStage] = field(default_factory=list)
    requirements: list[RawRequirement] = field(default_factory=list)
    penalties: list[RawPenalty] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def merge_chunks(extractions: list[ChunkExtraction]) -> MergedFacts:
    """Сводит разборы фрагментов в единый набор фактов."""
    merged = MergedFacts()
    if not extractions:
        return merged

    merged.subject = _vote_text([item.subject for item in extractions])
    merged.customer = _vote_text([item.customer for item in extractions])
    merged.procurement_number = _vote_text([item.procurement_number for item in extractions])
    merged.law = _vote_text([item.law for item in extractions if item.law != "не определено"])
    merged.duration_text = _first_longest([item.duration_text for item in extractions])

    merged.price, price_conflict = _merge_money(
        [item.price for item in extractions], "цену контракта"
    )
    merged.contract_security, security_conflict = _merge_money(
        [item.contract_security for item in extractions], "обеспечение исполнения контракта"
    )

    merged.application_deadline, deadline_conflict = _merge_date(
        [item.application_deadline for item in extractions], "дату окончания подачи заявок"
    )
    merged.contract_start, start_conflict = _merge_date(
        [item.contract_start for item in extractions], "дату начала исполнения"
    )
    merged.contract_end, end_conflict = _merge_date(
        [item.contract_end for item in extractions], "дату окончания исполнения"
    )

    merged.stages = _dedupe(
        [stage for item in extractions for stage in item.stages], key=lambda s: s.name
    )
    merged.requirements = _dedupe(
        [req for item in extractions for req in item.requirements], key=lambda r: r.text
    )
    merged.penalties = _dedupe(
        [pen for item in extractions for pen in item.penalties],
        key=lambda p: f"{p.kind} {p.trigger}",
    )

    merged.conflicts = [
        message
        for message in (
            price_conflict,
            security_conflict,
            deadline_conflict,
            start_conflict,
            end_conflict,
        )
        if message
    ]
    return merged


def _vote_text(values: list[str | None]) -> str | None:
    """Самое частое непустое значение; при равенстве — самое подробное."""
    candidates = [value.strip() for value in values if value and value.strip()]
    if not candidates:
        return None
    counter = Counter(candidates)
    best = counter.most_common(1)[0][1]
    tied = [value for value, count in counter.items() if count == best]
    return max(tied, key=len)


def _first_longest(values: list[str | None]) -> str | None:
    candidates = [value.strip() for value in values if value and value.strip()]
    return max(candidates, key=len) if candidates else None


def _merge_money(values: list[RawMoney | None], label: str) -> tuple[RawMoney | None, str | None]:
    """Выбирает сумму и сообщает, если фрагменты назвали разные."""
    candidates = [value for value in values if value and value.amount is not None]
    if not candidates:
        # Сумма могла прийти без числа, но с формулировкой — это лучше, чем ничего.
        fallback = next((value for value in values if value is not None), None)
        return fallback, None

    counter = Counter(candidate.amount for candidate in candidates)
    winner_amount = counter.most_common(1)[0][0]
    winner = next(
        (candidate for candidate in candidates if candidate.amount == winner_amount), candidates[0]
    )

    conflict = None
    if len(counter) > 1:
        listed = ", ".join(f"{amount:,.2f}".replace(",", " ") for amount in sorted(counter))
        conflict = (
            f"Фрагменты документа называют разные значения на {label}: {listed}. "
            f"Выбрано наиболее частое."
        )
    return winner, conflict


def _merge_date(values: list[RawDate | None], label: str) -> tuple[RawDate | None, str | None]:
    candidates = [value for value in values if value and (value.iso_date or value.raw_text)]
    if not candidates:
        return None, None

    dated = [value for value in candidates if value.iso_date]
    if not dated:
        return candidates[0], None

    counter = Counter(value.iso_date for value in dated)
    winner_date = counter.most_common(1)[0][0]
    winner = next(value for value in dated if value.iso_date == winner_date)

    conflict = None
    if len(counter) > 1:
        conflict = (
            f"Фрагменты документа называют разные значения на {label}: "
            f"{', '.join(sorted(str(item) for item in counter))}. Выбрано наиболее частое."
        )
    return winner, conflict


def _dedupe(items: list[T], key) -> list[T]:  # noqa: ANN001 — key возвращает строку из любого типа
    """Убирает повторы, приехавшие из перекрытия фрагментов.

    Точного совпадения строк недостаточно: одно и то же требование в двух
    фрагментах модель нередко формулирует чуть по-разному, а на границе
    перекрытия — обрезает. Поэтому сравнение нечёткое, и из пары похожих
    остаётся более подробная формулировка.
    """
    kept: list[T] = []
    keys: list[str] = []

    for item in items:
        candidate_key = normalize(str(key(item)))
        if not candidate_key:
            continue

        duplicate_index = None
        for index, existing in enumerate(keys):
            if _similar(candidate_key, existing):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(item)
            keys.append(candidate_key)
        elif len(candidate_key) > len(keys[duplicate_index]):
            kept[duplicate_index] = item
            keys[duplicate_index] = candidate_key

    return kept


def _similar(left: str, right: str) -> bool:
    if left == right:
        return True
    # Обрезанный на границе фрагмента текст — начало более полного.
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 40 and longer.startswith(shorter[:40]):
        return True
    if abs(len(left) - len(right)) > max(len(left), len(right)) * 0.5:
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= SIMILARITY_THRESHOLD
