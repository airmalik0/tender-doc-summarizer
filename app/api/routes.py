"""HTTP-эндпоинты сервиса."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_pipeline, get_provider, get_settings
from app.core.config import Settings
from app.core.exceptions import (
    FileTooLargeError,
    NotFoundError,
    PdfParseError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.summary import ErrorResponse, HealthResponse, TenderSummary
from app.services.extraction.pipeline import SummarizationPipeline
from app.services.llm.base import LLMProvider
from app.services.pdf.ocr import ocr_available

logger = get_logger(__name__)
router = APIRouter()

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"
PDF_MAGIC = b"%PDF-"
READ_CHUNK = 1024 * 1024


class SampleFile(BaseModel):
    """Демонстрационный документ, лежащий в поставке."""

    name: str = Field(description="Имя файла, его же передавать в /summarize/sample")
    size_kb: int
    title: str = Field(description="Человекочитаемое описание")


# Ошибки одинаковы у обоих разборов и описываются в схеме, чтобы Swagger
# показывал не только счастливый путь.
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Документ не найден"},
    413: {"model": ErrorResponse, "description": "Файл или документ больше лимита"},
    415: {"model": ErrorResponse, "description": "Неподдерживаемый тип файла"},
    422: {"model": ErrorResponse, "description": "PDF не разбирается или пуст"},
    502: {"model": ErrorResponse, "description": "Провайдер недоступен или ответил ошибкой"},
    503: {"model": ErrorResponse, "description": "Провайдер не настроен"},
}

SAMPLE_TITLES = {
    "tender-44fz-remont-krovli.pdf": (
        "Электронный аукцион: капремонт кровли школы (текстовый PDF)"
    ),
    "tender-44fz-postavka-oborudovaniya.pdf": (
        "Запрос котировок: поставка медоборудования (текстовый PDF)"
    ),
    "tender-scan-postavka-kanctovarov.pdf": "Извещение о закупке канцтоваров (скан, требует OCR)",
}


@router.get("/health", response_model=HealthResponse, summary="Состояние сервиса")
async def health(
    settings: Settings = Depends(get_settings),
    provider: LLMProvider = Depends(get_provider),
) -> HealthResponse:
    """Проверка живости и текущий режим работы.

    Наружу отдаётся ровно то, что нужно для диагностики: какой провайдер
    выбран, какая модель и доступен ли OCR. Ключи, разумеется, не отдаются.
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        provider=provider.name,
        model=provider.model,
        ocr_available=ocr_available(),
        max_upload_mb=settings.max_upload_mb,
        configured_providers=settings.configured_providers(),
    )


@router.get("/samples", response_model=list[SampleFile], summary="Демонстрационные документы")
async def list_samples() -> list[SampleFile]:
    """Список документов из папки samples, чтобы попробовать сервис без своего PDF."""
    if not SAMPLES_DIR.exists():
        return []
    return [
        SampleFile(
            name=path.name,
            size_kb=round(path.stat().st_size / 1024),
            title=SAMPLE_TITLES.get(path.name, path.stem),
        )
        for path in sorted(SAMPLES_DIR.glob("*.pdf"))
    ]


@router.post(
    "/summarize",
    response_model=TenderSummary,
    summary="Разобрать тендерную документацию",
    response_model_exclude_none=False,
    responses=ERROR_RESPONSES,
)
async def summarize(
    file: UploadFile = File(description="PDF с тендерной документацией"),
    use_cache: bool = Query(default=True, description="Брать готовый разбор из кэша, если он есть"),
    settings: Settings = Depends(get_settings),
    pipeline: SummarizationPipeline = Depends(get_pipeline),
) -> TenderSummary:
    """Принимает PDF и возвращает структурированную выжимку.

    Что происходит внутри: текст извлекается из PDF (при необходимости через
    OCR), документ режется на фрагменты, по каждому идёт извлечение фактов,
    результаты сливаются, цитаты проверяются по исходному тексту.
    """
    data = await _read_upload(file, settings.max_upload_bytes)
    filename = file.filename or "document.pdf"
    _ensure_pdf(data, filename)

    return await pipeline.run(data=data, filename=filename, use_cache=use_cache)


@router.post(
    "/summarize/sample",
    response_model=TenderSummary,
    summary="Разобрать демонстрационный документ",
    responses=ERROR_RESPONSES,
)
async def summarize_sample(
    name: str = Query(description="Имя файла из /samples"),
    use_cache: bool = Query(default=True),
    pipeline: SummarizationPipeline = Depends(get_pipeline),
) -> TenderSummary:
    """Разбирает документ из поставки — чтобы проверить сервис в один клик."""
    # Имя приходит снаружи: берём только базовую часть, иначе ../../etc/passwd
    # прочитает что угодно за пределами папки с примерами.
    safe_name = Path(name).name
    path = SAMPLES_DIR / safe_name

    if not path.exists() or path.suffix.lower() != ".pdf":
        raise NotFoundError(
            f"Демонстрационный документ «{safe_name}» не найден.",
            details={"available": [item.name for item in sorted(SAMPLES_DIR.glob("*.pdf"))]},
        )

    return await pipeline.run(data=path.read_bytes(), filename=safe_name, use_cache=use_cache)


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Читает загрузку кусками, останавливаясь на превышении лимита.

    Тело запроса к этому моменту уже принято: FastAPI разбирает multipart до
    вызова обработчика, и крупная часть уходит во временный файл на диске.
    Поэтому здесь ограничивается только то, что попадёт в память и в разбор,
    а раннее отсечение по Content-Length живёт в middleware приложения.
    """
    buffer = bytearray()
    while chunk := await file.read(READ_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise FileTooLargeError(
                f"Файл больше допустимых {limit // (1024 * 1024)} МБ.",
                details={"limit_bytes": limit},
            )
    return bytes(buffer)


def _ensure_pdf(data: bytes, filename: str) -> None:
    """Проверяет, что это действительно PDF."""
    if not data:
        raise UnsupportedFileTypeError("Файл пустой.")
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(
            "Ожидается файл с расширением .pdf.", details={"filename": filename}
        )
    # Расширению верить нельзя — проверяем сигнатуру.
    if not data.lstrip()[:8].startswith(PDF_MAGIC):
        raise PdfParseError(
            "Содержимое файла не похоже на PDF: отсутствует сигнатура %PDF-.",
            details={"filename": filename},
        )
