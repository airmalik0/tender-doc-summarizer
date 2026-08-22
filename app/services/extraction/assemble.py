"""Сборка итогового ответа и оценка его достоверности.

Здесь сходятся три источника: факты от модели, факты от детерминированного
парсера и сам текст документа. Каждая цитата проверяется поиском по тексту,
каждая ключевая сумма сверяется с найденной правилами, и по результатам
считается confidence.

Смысл шага в том, чтобы сервис никогда не отдавал уверенный на вид JSON,
за который никто не отвечает. Если проверка не сошлась — это видно в
ответе, а не остаётся внутри логов.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from app.models.extraction import (
    DocumentSynthesis,
    RawDate,
    RawEvidence,
    RawMoney,
    RawPenalty,
    RawRequirement,
    RawStage,
)
from app.models.summary import (
    DateValue,
    DocumentInfo,
    Evidence,
    MoneyValue,
    Penalty,
    ProcessingMeta,
    Requirement,
    Stage,
    TenderSummary,
    Timeline,
)
from app.services.extraction.heuristics import RuleFindings, parse_iso_date
from app.services.extraction.merge import MergedFacts
from app.services.extraction.verify import DocumentIndex

# Насколько сумма может отличаться от найденной правилами, чтобы считаться той же.
AMOUNT_TOLERANCE = Decimal("0.01")


class _EvidenceChecker:
    """Проверяет цитаты и считает, сколько из них не подтвердилось."""

    def __init__(self, index: DocumentIndex) -> None:
        self._index = index
        self.total = 0
        self.unverified = 0
        self.page_corrections = 0

    def convert(self, raw: RawEvidence | None) -> Evidence | None:
        if raw is None or raw.is_empty:
            return None

        self.total += 1
        check = self._index.check(raw.quote, raw.page)
        if not check.verified:
            self.unverified += 1
        elif check.page is not None and check.page != raw.page:
            self.page_corrections += 1

        return Evidence(
            page=check.page if check.page is not None else raw.page,
            quote=raw.quote.strip(),
            verified=check.verified,
        )


def assemble_summary(
    *,
    pages: list[tuple[int, str]],
    document_info: DocumentInfo,
    facts: MergedFacts,
    rules: RuleFindings,
    synthesis: DocumentSynthesis | None,
    meta: ProcessingMeta,
    warnings: list[str],
) -> TenderSummary:
    """Собирает ответ API из фактов, правил и текста документа."""
    checker = _EvidenceChecker(DocumentIndex(pages))
    collected: list[str] = list(warnings)

    price = _money(facts.price, checker, rules)
    contract_security = _money(facts.contract_security, checker, rules)

    timeline = Timeline(
        application_deadline=_date(facts.application_deadline, checker),
        contract_start=_date(facts.contract_start, checker),
        contract_end=_date(facts.contract_end, checker),
        duration_text=facts.duration_text,
        stages=[_stage(stage, checker) for stage in facts.stages],
    )
    requirements = [_requirement(item, checker) for item in facts.requirements]
    penalties = [_penalty(item, checker) for item in facts.penalties]

    collected.extend(facts.conflicts)
    collected.extend(
        _quality_warnings(
            price=price,
            contract_security=contract_security,
            timeline=timeline,
            requirements=requirements,
            penalties=penalties,
            checker=checker,
            rules=rules,
        )
    )

    law = facts.law if facts.law in {"44-ФЗ", "223-ФЗ"} else rules.law

    summary = TenderSummary(
        document=document_info,
        subject=facts.subject,
        customer=facts.customer,
        procurement_number=facts.procurement_number or rules.procurement_number,
        law=law,  # type: ignore[arg-type]
        price=price,
        contract_security=contract_security,
        timeline=timeline,
        requirements=requirements,
        penalties=penalties,
        summary=synthesis.summary if synthesis else "",
        risks=list(synthesis.risks) if synthesis else [],
        warnings=collected,
        meta=meta,
    )
    summary.confidence = compute_confidence(summary, checker=checker, offline=meta.provider == "offline")
    return summary


def compute_confidence(
    summary: TenderSummary, *, checker: _EvidenceChecker, offline: bool
) -> float:
    """Оценка достоверности разбора.

    Смысл шкалы: 1.0 — все ключевые поля заполнены, все цитаты нашлись в
    документе, сумма контракта совпала с найденной правилами. Дальше за
    каждый изъян вычитается фиксированная доля. Формула намеренно простая
    и прозрачная — её должно быть видно глазами в ответе, а не угадывать.
    """
    score = 1.0

    if summary.price is None or summary.price.amount is None:
        score -= 0.25
    else:
        if not summary.price.confirmed_by_rules:
            score -= 0.10
        if summary.price.evidence is not None and not summary.price.evidence.verified:
            score -= 0.10

    timeline = summary.timeline
    has_dates = any(
        value is not None and (value.date is not None or value.raw_text)
        for value in (timeline.application_deadline, timeline.contract_start, timeline.contract_end)
    )
    if not has_dates and not timeline.duration_text:
        score -= 0.15

    if not summary.requirements:
        score -= 0.10
    if not summary.penalties:
        score -= 0.10

    if checker.total:
        unverified_share = checker.unverified / checker.total
        score -= 0.25 * unverified_share

    if summary.document.ocr_pages:
        # Распознанный текст всегда чуть менее надёжен, чем текстовый слой.
        score -= 0.05

    if offline:
        # Правила не понимают контекст: потолок оценки принципиально ниже.
        score = min(score, 0.6)

    return round(max(0.0, min(1.0, score)), 2)


def _money(
    raw: RawMoney | None, checker: _EvidenceChecker, rules: RuleFindings
) -> MoneyValue | None:
    if raw is None:
        return None

    amount = _to_decimal(raw.amount if raw.amount > 0 else None)
    return MoneyValue(
        amount=amount,
        currency=(raw.currency or "RUB").upper(),
        vat=raw.vat,
        raw_text=raw.raw_text or None,
        evidence=checker.convert(raw.evidence),
        confirmed_by_rules=_confirmed(amount, rules),
    )


def _confirmed(amount: Decimal | None, rules: RuleFindings) -> bool:
    """Нашёл ли ровно такую сумму независимый детерминированный парсер."""
    if amount is None:
        return False
    return any(abs(amount - found) <= AMOUNT_TOLERANCE for found in rules.amounts)


def _to_decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _date(raw: RawDate | None, checker: _EvidenceChecker) -> DateValue | None:
    if raw is None:
        return None
    return DateValue(
        date=parse_iso_date(raw.iso_date) or parse_iso_date(raw.raw_text),
        raw_text=raw.raw_text or None,
        evidence=checker.convert(raw.evidence),
    )


def _stage(raw: RawStage, checker: _EvidenceChecker) -> Stage:
    return Stage(
        name=raw.name,
        deadline_text=raw.deadline_text or None,
        evidence=checker.convert(raw.evidence),
    )


def _requirement(raw: RawRequirement, checker: _EvidenceChecker) -> Requirement:
    return Requirement(
        text=raw.text,
        category=raw.category,
        mandatory=raw.mandatory,
        evidence=checker.convert(raw.evidence),
    )


def _penalty(raw: RawPenalty, checker: _EvidenceChecker) -> Penalty:
    return Penalty(
        kind=raw.kind,
        party=raw.party,
        trigger=raw.trigger,
        calculation=raw.calculation or None,
        amount_text=raw.amount_text or None,
        evidence=checker.convert(raw.evidence),
    )


def _quality_warnings(
    *,
    price: MoneyValue | None,
    contract_security: MoneyValue | None,
    timeline: Timeline,
    requirements: list[Requirement],
    penalties: list[Penalty],
    checker: _EvidenceChecker,
    rules: RuleFindings,
) -> list[str]:
    """Всё, на что человеку стоит взглянуть глазами."""
    messages: list[str] = []

    if price is None or price.amount is None:
        messages.append("Цена контракта не найдена — проверьте документ вручную.")
    else:
        if not price.confirmed_by_rules:
            messages.append(
                f"Цена контракта {price.amount} не подтверждена детерминированным парсером: "
                f"такой суммы нет среди найденных в тексте. Возможна ошибка извлечения."
            )
        if rules.price is not None and abs(price.amount - rules.price.amount) > AMOUNT_TOLERANCE:
            messages.append(
                f"Расхождение по цене контракта: модель называет {price.amount}, "
                f"парсер по ключевым формулировкам — {rules.price.amount}."
            )

    if contract_security is not None and contract_security.amount is not None:
        if not contract_security.confirmed_by_rules:
            messages.append(
                f"Обеспечение исполнения контракта {contract_security.amount} "
                f"не подтверждено детерминированным парсером."
            )

    if not requirements:
        messages.append("Требования к участникам не найдены.")
    if not penalties:
        messages.append("Меры ответственности не найдены — это редкость для закупочной документации.")
    if (
        timeline.application_deadline is None
        and timeline.contract_end is None
        and not timeline.duration_text
    ):
        messages.append("Сроки не найдены.")

    if checker.unverified:
        messages.append(
            f"Цитат не найдено в тексте документа: {checker.unverified} из {checker.total}. "
            f"Соответствующие значения помечены verified=false."
        )
    if checker.page_corrections:
        messages.append(
            f"Номер страницы исправлен по результатам поиска цитаты: "
            f"{checker.page_corrections} шт."
        )

    return messages
