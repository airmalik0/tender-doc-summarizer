"""Выжимка без модели — для offline-режима.

Связный текст по шаблону, собранный из уже извлечённых фактов. Никаких
попыток изобразить естественную речь: задача — чтобы человек за пять
секунд понял, что за закупка, даже когда ключа для LLM нет.
"""

from __future__ import annotations

from decimal import Decimal

from app.models.extraction import DocumentSynthesis
from app.services.extraction.merge import MergedFacts


def build_fallback_synthesis(facts: MergedFacts) -> DocumentSynthesis:
    """Собирает summary и risks по шаблону."""
    sentences: list[str] = []

    subject = (facts.subject or "").strip()
    customer = (facts.customer or "").strip()
    if subject and customer:
        sentences.append(f"Предмет закупки: {subject}. Заказчик: {customer}.")
    elif subject:
        sentences.append(f"Предмет закупки: {subject}.")
    elif customer:
        sentences.append(f"Заказчик: {customer}.")

    if facts.law and facts.law != "не определено":
        number = f" (извещение № {facts.procurement_number})" if facts.procurement_number else ""
        sentences.append(f"Закупка проводится по {facts.law}{number}.")

    if facts.price and facts.price.amount is not None:
        vat = f", НДС {facts.price.vat}" if facts.price.vat else ""
        sentences.append(
            f"Начальная (максимальная) цена контракта — {_money(facts.price.amount)}{vat}."
        )

    if facts.contract_security and facts.contract_security.amount is not None:
        sentences.append(
            f"Обеспечение исполнения контракта — {_money(facts.contract_security.amount)}."
        )

    if facts.duration_text:
        sentences.append(facts.duration_text.rstrip(".") + ".")
    if facts.application_deadline and facts.application_deadline.raw_text:
        sentences.append(f"Приём заявок завершается {facts.application_deadline.raw_text}.")

    if facts.requirements:
        sentences.append(f"Требований к участникам найдено: {len(facts.requirements)}.")
    if facts.penalties:
        sentences.append(f"Мер ответственности найдено: {len(facts.penalties)}.")

    return DocumentSynthesis(summary=" ".join(sentences), risks=_risks(facts))


def _risks(facts: MergedFacts) -> list[str]:
    risks: list[str] = []

    for penalty in facts.penalties:
        if penalty.kind == "штраф" and penalty.amount_text and penalty.party == "исполнитель":
            risks.append(f"Штраф для исполнителя: {penalty.amount_text} — {penalty.trigger[:120]}")
        if len(risks) >= 3:
            break

    if facts.contract_security and facts.contract_security.amount is not None:
        risks.append(
            f"На время исполнения контракта замораживается обеспечение "
            f"{_money(facts.contract_security.amount)}."
        )

    experience = [item for item in facts.requirements if item.category == "опыт"]
    if experience:
        risks.append(f"Барьер входа по опыту: {experience[0].text[:160]}")

    return risks[:5]


def _money(amount: float | Decimal) -> str:
    """1234567.89 → «1 234 567,89 руб.»"""
    formatted = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
    return f"{formatted} руб."
