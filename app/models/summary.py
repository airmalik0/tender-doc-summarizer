"""Схемы ответа API.

В отличие от app/models/extraction.py, здесь значения уже приведены к типам
(Decimal для денег, date для дат) и снабжены результатом проверки: подтверждена
ли цитата поиском по исходному тексту и совпала ли сумма с той, что нашёл
детерминированный парсер.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from pydantic import BaseModel, Field

from app.models.extraction import (
    LawKind,
    PenaltyKind,
    PenaltyParty,
    RequirementCategory,
    VatMode,
)


class Evidence(BaseModel):
    """Цитата-подтверждение с результатом сверки по исходному тексту."""

    page: int = Field(description="Номер страницы документа")
    quote: str = Field(description="Цитата из документа")
    verified: bool = Field(
        description=(
            "true — цитата найдена в тексте документа; false — не найдена, "
            "значение следует перепроверить вручную"
        )
    )


class MoneyValue(BaseModel):
    """Денежная величина."""

    amount: Decimal | None = Field(default=None, description="Сумма в рублях")
    currency: str = Field(default="RUB", description="Валюта")
    vat: VatMode = Field(default="не указано", description="Учёт НДС")
    raw_text: str | None = Field(default=None, description="Формулировка суммы из документа")
    evidence: Evidence | None = None
    confirmed_by_rules: bool = Field(
        default=False,
        description=(
            "true — ровно такая сумма найдена в документе детерминированным парсером, "
            "то есть значение подтверждено независимо от модели"
        ),
    )


class DateValue(BaseModel):
    """Срок: разобранная дата и/или исходная формулировка."""

    # Аннотация через dt.date: поле называется date и внутри класса затеняет имя типа.
    date: dt.date | None = Field(default=None, description="Дата в формате ГГГГ-ММ-ДД")
    raw_text: str | None = Field(default=None, description="Формулировка срока из документа")
    evidence: Evidence | None = None


class Stage(BaseModel):
    """Этап исполнения контракта."""

    name: str
    deadline_text: str | None = None
    evidence: Evidence | None = None


class Requirement(BaseModel):
    """Требование к исполнителю или к предмету закупки."""

    text: str
    category: RequirementCategory = "прочее"
    mandatory: bool = True
    evidence: Evidence | None = None


class Penalty(BaseModel):
    """Мера ответственности."""

    kind: PenaltyKind
    party: PenaltyParty = "не указано"
    trigger: str
    calculation: str | None = None
    amount_text: str | None = None
    evidence: Evidence | None = None


class Timeline(BaseModel):
    """Сроки по контракту."""

    application_deadline: DateValue | None = Field(
        default=None, description="Окончание подачи заявок"
    )
    contract_start: DateValue | None = Field(default=None, description="Начало исполнения")
    contract_end: DateValue | None = Field(default=None, description="Окончание исполнения")
    duration_text: str | None = Field(
        default=None, description="Срок исполнения целиком, формулировкой документа"
    )
    stages: list[Stage] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Что за файл разобрали."""

    filename: str
    sha256: str = Field(description="Хэш содержимого файла, он же ключ кэша")
    pages: int
    characters: int = Field(description="Сколько символов текста извлечено")
    text_source: str = Field(
        description="text_layer — текстовый слой, ocr — распознавание, mixed — и то и другое"
    )
    ocr_pages: list[int] = Field(
        default_factory=list, description="Страницы, распознанные через OCR"
    )


class ProcessingMeta(BaseModel):
    """Как именно получен результат — для отладки и оценки стоимости."""

    provider: str = Field(description="anthropic | openai | gemini | offline")
    model: str | None = None
    chunks: int = Field(description="На сколько фрагментов был разбит документ")
    llm_calls: int = Field(description="Сколько запросов ушло в модель")
    input_tokens: int | None = None
    output_tokens: int | None = None
    pdf_ms: int = Field(description="Время разбора PDF, мс")
    llm_ms: int = Field(description="Время работы модели, мс")
    total_ms: int
    cached: bool = Field(default=False, description="Ответ отдан из кэша")


class TenderSummary(BaseModel):
    """Итоговая выжимка по тендерной документации."""

    document: DocumentInfo
    subject: str | None = Field(default=None, description="Предмет закупки")
    customer: str | None = Field(default=None, description="Заказчик")
    procurement_number: str | None = Field(default=None, description="Номер извещения")
    law: LawKind = Field(default="не определено")

    price: MoneyValue | None = Field(
        default=None, description="Начальная (максимальная) цена контракта"
    )
    contract_security: MoneyValue | None = Field(
        default=None, description="Обеспечение исполнения контракта"
    )
    timeline: Timeline = Field(default_factory=Timeline)
    requirements: list[Requirement] = Field(default_factory=list)
    penalties: list[Penalty] = Field(default_factory=list)

    summary: str = Field(default="", description="Краткая выжимка по документу")
    risks: list[str] = Field(default_factory=list, description="Риски для исполнителя")

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Оценка достоверности разбора: снижается за неподтверждённые цитаты, "
            "расхождение с детерминированным парсером и незаполненные ключевые поля"
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Всё, на что стоит посмотреть глазами: понижения режима, расхождения, пропуски",
    )
    meta: ProcessingMeta


class ErrorResponse(BaseModel):
    """Единый формат ошибки."""

    code: str = Field(description="Машиночитаемый код ошибки")
    message: str = Field(description="Человекочитаемое описание")
    details: dict[str, object] = Field(default_factory=dict)
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Состояние сервиса."""

    status: str
    version: str
    provider: str = Field(description="Провайдер, который будет использован при разборе")
    model: str | None = None
    ocr_available: bool
    configured_providers: list[str] = Field(
        default_factory=list, description="Провайдеры, для которых задан ключ"
    )
