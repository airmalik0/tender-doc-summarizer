"""Тесты нарезки документа на фрагменты."""

from __future__ import annotations

from app.core.config import Settings
from app.services.extraction.chunking import build_chunks
from app.services.pdf.extractor import DocumentText, PageText


def _document(*texts: str) -> DocumentText:
    return DocumentText(
        pages=[
            PageText(number=index + 1, text=text, source="text_layer")
            for index, text in enumerate(texts)
        ]
    )


def test_short_document_is_single_chunk() -> None:
    chunks, warnings = build_chunks(_document("текст первой", "текст второй"), Settings())

    assert len(chunks) == 1
    assert chunks[0].pages == [1, 2]
    assert warnings == []


def test_every_chunk_starts_with_page_marker() -> None:
    chunks, _ = build_chunks(_document("а" * 900, "б" * 900, "в" * 900), Settings(chunk_chars=1000))

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.text.startswith("[СТРАНИЦА ")


def test_overlap_carries_its_page_marker() -> None:
    """Хвост предыдущего фрагмента без маркера сделал бы номер страницы враньём."""
    chunks, _ = build_chunks(
        _document("первая. " * 120, "вторая. " * 120),
        Settings(chunk_chars=1000, chunk_overlap_chars=200),
    )

    assert len(chunks) == 2
    assert chunks[1].text.startswith("[СТРАНИЦА 1]")
    assert "продолжение предыдущего фрагмента" in chunks[1].text


def test_long_page_is_split_by_paragraphs() -> None:
    page = "\n\n".join(["абзац " * 60] * 12)
    chunks, _ = build_chunks(_document(page), Settings(chunk_chars=1000, chunk_overlap_chars=0))

    assert len(chunks) > 1
    assert all(chunk.pages == [1] for chunk in chunks)


def test_chunk_limit_produces_warning() -> None:
    pages = ["страница " * 200 for _ in range(10)]
    chunks, warnings = build_chunks(
        _document(*pages), Settings(chunk_chars=1000, chunk_overlap_chars=0, max_chunks=3)
    )

    assert len(chunks) == 3
    assert warnings and "MAX_CHUNKS" in warnings[0]


def test_empty_pages_are_skipped() -> None:
    chunks, _ = build_chunks(_document("содержимое", "   ", "ещё содержимое"), Settings())
    assert chunks[0].pages == [1, 3]


def test_pages_label_is_human_readable() -> None:
    chunks, _ = build_chunks(_document("а", "б", "в"), Settings())
    assert chunks[0].pages_label == "1–3"
