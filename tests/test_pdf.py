"""Тесты разбора PDF."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import DocumentTooLongError, EmptyDocumentError, PdfParseError
from app.services.pdf.extractor import extract_document, normalize_text


def test_text_layer_is_extracted(roof_pdf: bytes, settings: Settings) -> None:
    document = extract_document(roof_pdf, settings)

    assert document.page_count == 4
    assert document.source_summary == "text_layer"
    assert document.ocr_pages == []
    assert "12 480 350,00" in document.pages[0].text


def test_page_markers_are_present_in_prompt_text(roof_pdf: bytes, settings: Settings) -> None:
    """Без маркеров страниц модель не сможет сослаться на страницу в цитате."""
    text = extract_document(roof_pdf, settings).as_prompt_text()
    assert "[СТРАНИЦА 1]" in text
    assert "[СТРАНИЦА 4]" in text


def test_scan_without_ocr_raises(scan_pdf: bytes, settings: Settings) -> None:
    """У скана нет текстового слоя: с выключенным OCR это честная 4xx-ошибка."""
    with pytest.raises(EmptyDocumentError):
        extract_document(scan_pdf, settings)


def test_broken_file_raises_parse_error(settings: Settings) -> None:
    with pytest.raises(PdfParseError):
        extract_document("%PDF-1.7\nэто не настоящий pdf".encode(), settings)


def test_page_limit_is_enforced(roof_pdf: bytes) -> None:
    with pytest.raises(DocumentTooLongError):
        extract_document(roof_pdf, Settings(max_pages=1, ocr_enabled=False))


def test_hyphenated_linebreaks_are_joined() -> None:
    assert normalize_text("выпол-\nнение работ") == "выполнение работ"


def test_soft_hyphens_are_removed() -> None:
    assert normalize_text("догово­ра") == "договора"


def test_repeated_blank_lines_collapse() -> None:
    assert normalize_text("а\n\n\n\n\nб") == "а\n\nб"


def test_text_volume_limit_is_enforced(roof_pdf: bytes) -> None:
    """Предел на извлечённый текст, а не на размер файла.

    Страница с потоком, который распаковывается в десятки миллионов символов,
    весит килобайты и проходит любой лимит загрузки. Число символов известно
    до материализации строки, поэтому проверка стоит именно там.
    """
    with pytest.raises(DocumentTooLongError):
        extract_document(
            roof_pdf, Settings(max_document_chars=5_000, ocr_enabled=False, _env_file=None)
        )


def test_render_scale_is_clamped_by_area() -> None:
    """Страница с огромным MediaBox иначе даёт битмап на сотни мегабайт."""
    from app.services.pdf.extractor import _safe_scale

    class _HugePage:
        def get_size(self) -> tuple[float, float]:
            return 14400.0, 14400.0

    scale = _safe_scale(_HugePage(), dpi=600, max_pixels=40_000_000)
    assert (14400 * scale) * (14400 * scale) <= 40_000_000 * 1.01
