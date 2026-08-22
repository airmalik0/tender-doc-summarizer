#!/usr/bin/env python3
"""Генератор демонстрационных PDF для samples/.

Скрипт нужен, чтобы демо-документы были воспроизводимы: их можно
перегенерировать, поправить и не тащить в репозиторий непонятно откуда
взявшиеся бинарники. Один из документов дополнительно растрируется —
получается PDF без текстового слоя, на котором видно работу OCR-ветки.

    python scripts/make_samples.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_content import DOCUMENTS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples"

# Кириллический шрифт ищем среди системных: macOS для локального прогона,
# DejaVu/Liberation — для контейнеров и CI.
FONT_CANDIDATES: list[tuple[str, str]] = [
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
    (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ),
]


def register_fonts() -> tuple[str, str]:
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists():
            pdfmetrics.registerFont(TTFont("DocFont", regular))
            bold_name = "DocFont"
            if Path(bold).exists() and bold != regular:
                pdfmetrics.registerFont(TTFont("DocFont-Bold", bold))
                bold_name = "DocFont-Bold"
            pdfmetrics.registerFontFamily("DocFont", normal="DocFont", bold=bold_name)
            return "DocFont", bold_name
    raise SystemExit(
        "Не найден ни один кириллический TTF-шрифт. Установите fonts-dejavu "
        "или поправьте FONT_CANDIDATES."
    )


def build_styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    return {
        "h1": ParagraphStyle(
            "h1", fontName=bold, fontSize=14, leading=18, alignment=TA_CENTER, spaceAfter=8
        ),
        "h2": ParagraphStyle(
            "h2", fontName=bold, fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=6
        ),
        "center": ParagraphStyle(
            "center", fontName=font, fontSize=10.5, leading=14, alignment=TA_CENTER, spaceAfter=4
        ),
        "p": ParagraphStyle(
            "p", fontName=font, fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6
        ),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=9.5, leading=12.5),
        "cell_key": ParagraphStyle("cell_key", fontName=bold, fontSize=9.5, leading=12.5),
        "li": ParagraphStyle(
            "li", fontName=font, fontSize=10, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=4
        ),
    }


def build_flowables(blocks: list[tuple[str, object]], styles: dict[str, ParagraphStyle]) -> list:
    flowables: list = []
    for kind, payload in blocks:
        if kind in {"h1", "h2", "center", "p"}:
            flowables.append(Paragraph(str(payload), styles[kind]))
        elif kind == "spacer":
            flowables.append(Spacer(1, float(payload)))  # type: ignore[arg-type]
        elif kind == "pagebreak":
            flowables.append(PageBreak())
        elif kind == "kv":
            rows = [
                [Paragraph(key, styles["cell_key"]), Paragraph(value, styles["cell"])]
                for key, value in payload  # type: ignore[union-attr]
            ]
            table = Table(rows, colWidths=[58 * mm, 105 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F2F2")),
                    ]
                )
            )
            flowables.append(table)
        elif kind in {"ul", "ol"}:
            items = [
                ListItem(Paragraph(str(text), styles["li"]), leftIndent=14)
                for text in payload  # type: ignore[union-attr]
            ]
            flowables.append(
                ListFlowable(
                    items,
                    bulletType="bullet" if kind == "ul" else "1",
                    bulletFontName=styles["li"].fontName,
                    bulletFontSize=9,
                    leftIndent=16,
                )
            )
            flowables.append(Spacer(1, 4))
        else:  # pragma: no cover — опечатка в описании документа
            raise ValueError(f"Неизвестный тип блока: {kind}")
    return flowables


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DocFont", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"— {doc.page} —")
    canvas.drawString(
        20 * mm, 12 * mm, "Единая информационная система в сфере закупок · zakupki.gov.ru"
    )
    canvas.restoreState()


def render_pdf(document: dict[str, object], styles: dict[str, ParagraphStyle]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=str(document["title"]),
        author="Единая информационная система в сфере закупок",
    )
    doc.build(
        build_flowables(document["blocks"], styles),  # type: ignore[arg-type]
        onFirstPage=draw_footer,
        onLaterPages=draw_footer,
    )
    return buffer.getvalue()


def rasterize(pdf_bytes: bytes, dpi: int = 150) -> bytes:
    """Превращает PDF в набор картинок — получается «скан» без текстового слоя.

    Дополнительно добавляем лёгкий поворот и шум: реальный скан никогда не
    бывает идеально ровным, и OCR-ветка должна проверяться на похожем входе.
    """
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter

    pdf = pdfium.PdfDocument(pdf_bytes)
    images: list[Image.Image] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            image = page.render(scale=dpi / 72).to_pil().convert("L")
            image = image.rotate(0.35, resample=Image.BICUBIC, fillcolor=255, expand=False)
            image = image.filter(ImageFilter.GaussianBlur(0.3))
            images.append(image)
            page.close()
    finally:
        pdf.close()

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=float(dpi),
    )
    return buffer.getvalue()


def main() -> None:
    font, bold = register_fonts()
    styles = build_styles(font, bold)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for document in DOCUMENTS:
        pdf_bytes = render_pdf(document, styles)
        if document.get("rasterize"):
            pdf_bytes = rasterize(pdf_bytes)
        target = OUT_DIR / str(document["filename"])
        target.write_bytes(pdf_bytes)
        print(f"  {target.relative_to(ROOT)} — {len(pdf_bytes) / 1024:.0f} КБ")


if __name__ == "__main__":
    print("Генерирую демонстрационные документы:")
    main()
