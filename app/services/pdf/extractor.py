"""Извлечение текста из PDF.

Стратегия постраничная: сначала пробуем текстовый слой (быстро и точно),
и только если на странице его практически нет — рендерим её в картинку и
отправляем в OCR. Документы с сайтов госзакупок регулярно приходят
«гибридными»: часть страниц набрана, часть вклеена сканом, поэтому решение
принимается для каждой страницы отдельно, а не для файла целиком.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

import pypdfium2 as pdfium

from app.core.config import Settings
from app.core.exceptions import DocumentTooLongError, EmptyDocumentError, PdfParseError
from app.core.logging import get_logger
from app.services.pdf import ocr as ocr_backend

logger = get_logger(__name__)

PageSource = Literal["text_layer", "ocr", "empty"]

_SOFT_HYPHEN = "­"
_HYPHEN_LINEBREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_SPACES = re.compile(r"[ \t  ]{2,}")


@dataclass(slots=True)
class PageText:
    """Текст одной страницы и способ, которым он получен."""

    number: int  # нумерация с единицы, как её видит человек в просмотрщике
    text: str
    source: PageSource

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class DocumentText:
    """Постранично разобранный документ."""

    pages: list[PageText] = field(default_factory=list)
    extract_ms: int = 0

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def ocr_pages(self) -> list[int]:
        return [page.number for page in self.pages if page.source == "ocr"]

    @property
    def source_summary(self) -> str:
        kinds = {page.source for page in self.pages if page.source != "empty"}
        if not kinds:
            return "empty"
        if kinds == {"text_layer"}:
            return "text_layer"
        if kinds == {"ocr"}:
            return "ocr"
        return "mixed"

    def page(self, number: int) -> PageText | None:
        for page in self.pages:
            if page.number == number:
                return page
        return None

    def as_prompt_text(self) -> str:
        """Склеивает страницы с маркерами для промпта.

        Маркер страницы — единственный способ, которым модель может сослаться
        на конкретную страницу в evidence; без него цитата не проверяема.
        """
        return "\n\n".join(
            f"[СТРАНИЦА {page.number}]\n{page.text}" for page in self.pages if page.text.strip()
        )


def normalize_text(raw: str) -> str:
    """Приводит текст страницы к виду, пригодному и для модели, и для регексов."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace(_SOFT_HYPHEN, "")
    # «выпол-\nнение» → «выполнение»: иначе слово не найдётся ни поиском, ни моделью.
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("\n", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def extract_document(data: bytes, settings: Settings) -> DocumentText:
    """Разбирает PDF постранично. Блокирующая операция — вызывать в потоке."""
    started = time.perf_counter()

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise PdfParseError(
            "Не удалось открыть файл как PDF. Возможно, он повреждён или защищён паролем.",
            details={"reason": str(exc)},
        ) from exc

    try:
        page_count = len(pdf)
        if page_count == 0:
            raise PdfParseError("В PDF нет ни одной страницы.")
        if page_count > settings.max_pages:
            raise DocumentTooLongError(
                f"В документе {page_count} страниц, максимум — {settings.max_pages}.",
                details={"pages": page_count, "max_pages": settings.max_pages},
            )

        ocr_lang = ocr_backend.resolve_lang(settings.ocr_lang) if settings.ocr_enabled else ""
        ocr_ready = settings.ocr_enabled and ocr_backend.ocr_available()
        if settings.ocr_enabled and not ocr_ready:
            logger.warning("OCR включён в конфиге, но tesseract недоступен — сканы не распознаются")

        pages: list[PageText] = []
        total_chars = 0
        for index in range(page_count):
            page = pdf[index]
            text = normalize_text(_read_text_layer(page, settings.max_document_chars))
            source: PageSource = "text_layer" if text else "empty"

            if len(text) < settings.ocr_min_chars_per_page and ocr_ready:
                recognized = normalize_text(
                    _read_via_ocr(page, settings.ocr_dpi, ocr_lang, settings.max_render_pixels)
                )
                if len(recognized) > len(text):
                    text, source = recognized, "ocr"

            pages.append(PageText(number=index + 1, text=text, source=source))
            page.close()

            total_chars += len(text)
            if total_chars > settings.max_document_chars:
                raise DocumentTooLongError(
                    f"Из документа извлечено больше {settings.max_document_chars} символов "
                    f"текста — разбор прекращён на странице {index + 1}.",
                    details={"characters": total_chars, "limit": settings.max_document_chars},
                )
    finally:
        pdf.close()

    document = DocumentText(pages=pages, extract_ms=int((time.perf_counter() - started) * 1000))

    if document.total_chars == 0:
        raise EmptyDocumentError(
            "Из документа не удалось извлечь ни одного символа: нет текстового слоя, "
            "а OCR отключён или не смог распознать страницы.",
            details={"pages": document.page_count},
        )

    logger.info(
        "PDF разобран: страниц %d, символов %d, источник %s, OCR на страницах %s, %d мс",
        document.page_count,
        document.total_chars,
        document.source_summary,
        document.ocr_pages or "—",
        document.extract_ms,
    )
    return document


def _safe_scale(page: pdfium.PdfPage, dpi: int, max_pixels: int) -> float:
    """Масштаб рендера, ограниченный площадью растра.

    Страница с огромным MediaBox при обычном dpi даёт битмап на сотни мегабайт;
    файл при этом весит килобайты и проходит любой лимит загрузки.
    """
    scale = dpi / 72
    width, height = page.get_size()
    pixels = (width * scale) * (height * scale)
    if pixels > max_pixels and pixels > 0:
        scale *= (max_pixels / pixels) ** 0.5
        logger.warning("Страница слишком велика для рендера, масштаб снижен до %.2f", scale)
    return scale


def _read_text_layer(page: pdfium.PdfPage, max_chars: int) -> str:
    try:
        textpage = page.get_textpage()
    except Exception as exc:  # pragma: no cover — битая страница внутри валидного PDF
        logger.warning("Не удалось прочитать текстовый слой страницы: %s", exc)
        return ""
    try:
        # Сначала количество символов, потом сам текст: на странице, набитой
        # мусором, материализация строки удвоила бы и без того огромный расход
        # памяти, а число символов известно сразу и стоит бесплатно.
        count = textpage.count_chars()
        if count > max_chars:
            raise DocumentTooLongError(
                f"На одной странице {count} символов текста — это больше предела "
                f"{max_chars}. Похоже на намеренно раздутый документ.",
                details={"characters": count, "limit": max_chars},
            )
        return textpage.get_text_bounded()
    finally:
        textpage.close()


def _read_via_ocr(page: pdfium.PdfPage, dpi: int, lang: str, max_pixels: int) -> str:
    try:
        scale = _safe_scale(page, dpi, max_pixels)
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
    except Exception as exc:  # pragma: no cover — зависит от содержимого страницы
        logger.warning("Не удалось отрендерить страницу для OCR: %s", exc)
        return ""
    try:
        return ocr_backend.recognize(image, lang)
    except Exception as exc:  # pragma: no cover — зависит от окружения
        logger.warning("OCR страницы не удался: %s", exc)
        return ""
    finally:
        image.close()
