"""Разбиение документа на фрагменты для модели.

Аукционная документация бывает и на 4 страницы, и на 200. Целиком такой
документ либо не помещается в контекст, либо помещается, но качество
извлечения на длинном тексте заметно падает: модель начинает терять
детали из середины. Поэтому — map-reduce: извлечение по фрагментам,
затем слияние.

Границы фрагментов проходят по страницам, а не по символам: маркер
[СТРАНИЦА N] должен оставаться корректным, иначе номер страницы в цитате
станет враньём. Страница, не влезающая во фрагмент целиком, режется по
абзацам, и каждый кусок получает свой маркер.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import Settings
from app.services.pdf.extractor import DocumentText


@dataclass(slots=True)
class Chunk:
    """Фрагмент документа, уходящий в модель одним запросом."""

    index: int
    text: str
    pages: list[int] = field(default_factory=list)

    @property
    def pages_label(self) -> str:
        if not self.pages:
            return "—"
        if len(self.pages) == 1:
            return str(self.pages[0])
        return f"{min(self.pages)}–{max(self.pages)}"


def build_chunks(document: DocumentText, settings: Settings) -> tuple[list[Chunk], list[str]]:
    """Режет документ на фрагменты. Возвращает фрагменты и предупреждения."""
    warnings: list[str] = []
    blocks = _page_blocks(document, settings.chunk_chars)
    # Перекрытие больше четверти фрагмента лишено смысла: оно начинает
    # раздувать каждый запрос копией предыдущего.
    overlap = min(settings.chunk_overlap_chars, settings.chunk_chars // 4)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_pages: list[int] = []
    current_length = 0

    for page_number, block_text in blocks:
        block = f"[СТРАНИЦА {page_number}]\n{block_text}"
        if current_parts and current_length + len(block) > settings.chunk_chars:
            chunks.append(_make_chunk(len(chunks), current_parts, current_pages))
            tail = _overlap_tail(current_parts[-1], overlap)
            current_parts = [tail] if tail else []
            current_pages = [current_pages[-1]] if tail else []
            current_length = len(tail)

        current_parts.append(block)
        current_pages.append(page_number)
        current_length += len(block)

    if current_parts:
        chunks.append(_make_chunk(len(chunks), current_parts, current_pages))

    if len(chunks) > settings.max_chunks:
        dropped = len(chunks) - settings.max_chunks
        warnings.append(
            f"Документ разбит на {len(chunks)} фрагментов, обработаны первые "
            f"{settings.max_chunks}. Не разобрано фрагментов: {dropped}. "
            f"Увеличьте MAX_CHUNKS, если нужен полный разбор."
        )
        chunks = chunks[: settings.max_chunks]

    return chunks, warnings


def _page_blocks(document: DocumentText, limit: int) -> list[tuple[int, str]]:
    """Страницы, при необходимости разрезанные на части по абзацам."""
    blocks: list[tuple[int, str]] = []
    for page in document.pages:
        text = page.text.strip()
        if not text:
            continue
        if len(text) <= limit:
            blocks.append((page.number, text))
            continue
        for part in _split_long_text(text, limit):
            blocks.append((page.number, part))
    return blocks


def _split_long_text(text: str, limit: int) -> list[str]:
    """Режет слишком длинную страницу по абзацам, а совсем длинный абзац — по символам."""
    parts: list[str] = []
    buffer: list[str] = []
    length = 0

    for paragraph in text.split("\n\n"):
        if len(paragraph) > limit:
            if buffer:
                parts.append("\n\n".join(buffer))
                buffer, length = [], 0
            parts.extend(
                paragraph[offset : offset + limit] for offset in range(0, len(paragraph), limit)
            )
            continue
        if length + len(paragraph) > limit and buffer:
            parts.append("\n\n".join(buffer))
            buffer, length = [], 0
        buffer.append(paragraph)
        length += len(paragraph) + 2

    if buffer:
        parts.append("\n\n".join(buffer))
    return parts


def _overlap_tail(last_block: str, overlap: int) -> str:
    """Хвост предыдущего фрагмента, чтобы факт на стыке не потерялся.

    Хвост уносит с собой маркер своей страницы: без него модель не сможет
    правильно сослаться на страницу в цитате.
    """
    if overlap <= 0:
        return ""

    header, _, body = last_block.partition("\n")
    if len(body) <= overlap:
        return last_block

    tail = body[-overlap:]
    # Обрезаем до начала предложения, чтобы фрагмент не начинался с середины слова.
    cut = tail.find(". ")
    if 0 <= cut < overlap // 2:
        tail = tail[cut + 2 :]
    return f"{header}\n(продолжение предыдущего фрагмента)\n{tail}"


def _make_chunk(index: int, parts: list[str], pages: list[int]) -> Chunk:
    return Chunk(index=index, text="\n\n".join(parts), pages=sorted(set(pages)))
