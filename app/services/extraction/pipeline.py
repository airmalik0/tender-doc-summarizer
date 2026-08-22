"""Оркестратор разбора документа.

Порядок шагов и причины именно такого порядка:

1. Хэш содержимого и проверка кэша — до всякой работы, чтобы повторная
   загрузка того же файла не стоила ни секунды, ни токена.
2. Разбор PDF в отдельном потоке: pypdfium2 и tesseract блокирующие, а
   держать ими цикл событий нельзя — сервис перестанет отвечать остальным.
3. Детерминированный разбор правилами — по всему тексту сразу.
4. Нарезка на фрагменты и параллельное извлечение по ним (map).
5. Слияние (reduce) и синтез итоговой выжимки.
6. Сборка ответа с проверкой цитат и расчётом достоверности.

Отказ на отдельном фрагменте не роняет разбор: результат собирается из
уцелевших, а потеря отмечается в warnings. Сдаться целиком имеет смысл,
только если не уцелел ни один.
"""

from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMResponseInvalidError
from app.core.logging import get_logger
from app.models.extraction import ChunkExtraction, DocumentSynthesis
from app.models.summary import DocumentInfo, ProcessingMeta, TenderSummary
from app.services.cache import SummaryCache, content_hash
from app.services.extraction import heuristics
from app.services.extraction.assemble import assemble_summary
from app.services.extraction.chunking import Chunk, build_chunks
from app.services.extraction.fallback_summary import build_fallback_synthesis
from app.services.extraction.merge import MergedFacts, merge_chunks
from app.services.extraction.prompts import (
    EXTRACTION_SYSTEM,
    SYNTHESIS_SYSTEM,
    build_extraction_prompt,
    build_synthesis_prompt,
)
from app.services.llm.base import LLMProvider, LLMUsage
from app.services.pdf.extractor import extract_document

logger = get_logger(__name__)


def _describe(error: BaseException) -> str:
    """Человекочитаемая причина отказа вместе с деталями провайдера."""
    message = getattr(error, "message", None) or str(error)
    details = getattr(error, "details", None)
    reason = details.get("reason") if isinstance(details, dict) else None
    return f"{message} {reason}".strip() if reason else message


# Больше четырёх параллельных запросов к провайдеру смысла не имеют:
# упираемся в rate limit, а не в скорость.
MAX_PARALLEL_CHUNKS = 4


class SummarizationPipeline:
    """Полный путь от байтов PDF до готовой выжимки."""

    def __init__(self, settings: Settings, provider: LLMProvider, cache: SummaryCache) -> None:
        self._settings = settings
        self._provider = provider
        self._cache = cache

    async def run(self, *, data: bytes, filename: str, use_cache: bool = True) -> TenderSummary:
        started = time.perf_counter()
        sha256 = content_hash(data)
        cache_key = self._cache.key(sha256, self._provider.name, self._provider.model)

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Разбор %s взят из кэша", filename)
                # Кэш ищется по содержимому: тот же документ мог приехать под
                # другим именем, и отдавать чужое имя файла в ответе нечестно.
                cached.document.filename = filename
                cached.meta.cached = True
                cached.meta.total_ms = int((time.perf_counter() - started) * 1000)
                return cached

        # pypdfium2 и tesseract блокируют поток — уводим их с цикла событий.
        document = await asyncio.to_thread(extract_document, data, self._settings)
        pages = [(page.number, page.text) for page in document.pages]
        rules = heuristics.analyze(pages)

        chunks, warnings = build_chunks(document, self._settings)
        logger.info(
            "Документ %s: %d страниц, %d фрагментов, провайдер %s",
            filename,
            document.page_count,
            len(chunks),
            self._provider.name,
        )

        llm_started = time.perf_counter()
        extractions, usage, calls, chunk_warnings = await self._extract_chunks(chunks, filename)
        warnings.extend(chunk_warnings)

        facts = merge_chunks(extractions)
        synthesis, synthesis_usage, synthesis_calls, synthesis_warnings = await self._synthesize(
            facts, filename
        )
        usage = usage + synthesis_usage
        calls += synthesis_calls
        warnings.extend(synthesis_warnings)
        llm_ms = int((time.perf_counter() - llm_started) * 1000)

        if self._provider.name == "offline":
            warnings.insert(
                0,
                "Разбор выполнен без LLM, на детерминированных правилах: качество ниже "
                "полного режима. Задайте API-ключ в .env, чтобы включить модель.",
            )
        if document.ocr_pages:
            warnings.append(
                f"Часть страниц распознана через OCR ({', '.join(map(str, document.ocr_pages))}): "
                f"в тексте возможны искажения."
            )

        summary = assemble_summary(
            pages=pages,
            document_info=DocumentInfo(
                filename=filename,
                sha256=sha256,
                pages=document.page_count,
                characters=document.total_chars,
                text_source=document.source_summary,
                ocr_pages=document.ocr_pages,
            ),
            facts=facts,
            rules=rules,
            synthesis=synthesis,
            meta=ProcessingMeta(
                provider=self._provider.name,
                model=self._provider.model,
                chunks=len(chunks),
                llm_calls=calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                pdf_ms=document.extract_ms,
                llm_ms=llm_ms,
                total_ms=int((time.perf_counter() - started) * 1000),
            ),
            warnings=warnings,
        )

        self._cache.put(cache_key, summary)
        logger.info(
            "Разбор %s завершён за %d мс, достоверность %.2f",
            filename,
            summary.meta.total_ms,
            summary.confidence,
        )
        return summary

    async def _extract_chunks(
        self, chunks: list[Chunk], filename: str
    ) -> tuple[list[ChunkExtraction], LLMUsage, int, list[str]]:
        """Шаг map: параллельное извлечение по фрагментам."""
        semaphore = asyncio.Semaphore(min(MAX_PARALLEL_CHUNKS, max(1, len(chunks))))

        async def worker(chunk: Chunk) -> tuple[ChunkExtraction, LLMUsage]:
            async with semaphore:
                result = await self._provider.complete_json(
                    system=EXTRACTION_SYSTEM,
                    user=build_extraction_prompt(
                        chunk_text=chunk.text,
                        chunk_index=chunk.index,
                        chunk_total=len(chunks),
                        pages_label=chunk.pages_label,
                        filename=filename,
                    ),
                    schema_model=ChunkExtraction,
                )
            try:
                return ChunkExtraction.model_validate(result.data), result.usage
            except ValidationError as exc:
                raise LLMResponseInvalidError(
                    "Ответ модели не соответствует схеме извлечения.",
                    details={"errors": exc.errors(include_url=False)[:5]},
                ) from exc

        outcomes = await asyncio.gather(
            *(worker(chunk) for chunk in chunks), return_exceptions=True
        )

        extractions: list[ChunkExtraction] = []
        usage = LLMUsage()
        warnings: list[str] = []
        failures: list[str] = []
        calls = 0

        for chunk, outcome in zip(chunks, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                # Причину забираем вместе с деталями провайдера: без них и в
                # логе, и в ответе остаётся бессодержательное «не получилось».
                reason = _describe(outcome)
                failures.append(reason)
                logger.warning(
                    "Фрагмент %d (стр. %s) не разобран: %s",
                    chunk.index + 1,
                    chunk.pages_label,
                    reason,
                )
                warnings.append(
                    f"Фрагмент {chunk.index + 1} (страницы {chunk.pages_label}) не разобран: "
                    f"{reason}"
                )
                continue
            extraction, chunk_usage = outcome
            extractions.append(extraction)
            usage = usage + chunk_usage
            calls += 1

        if chunks and not extractions:
            # Уцелевших нет — отдавать пустой разбор нечестно. Причину первого
            # отказа обязательно прокидываем наружу: без неё клиент видит голое
            # «не получилось» и не может понять, дело в ключе, в сети или в
            # исчерпанном балансе.
            raise LLMError(
                (
                    f"Не удалось разобрать ни один фрагмент документа. {failures[0]}"
                    if failures
                    else "Не удалось разобрать ни один фрагмент документа."
                ),
                details={
                    "chunks": len(chunks),
                    "provider": self._provider.name,
                    "reason": failures[0] if failures else None,
                },
            )

        return extractions, usage, calls, warnings

    async def _synthesize(
        self, facts: MergedFacts, filename: str
    ) -> tuple[DocumentSynthesis, LLMUsage, int, list[str]]:
        """Шаг reduce: связная выжимка и риски по собранным фактам."""
        if not self._provider.supports_synthesis:
            return build_fallback_synthesis(facts), LLMUsage(), 0, []

        payload = {
            "предмет": facts.subject,
            "заказчик": facts.customer,
            "номер_извещения": facts.procurement_number,
            "закон": facts.law,
            "цена": facts.price.model_dump(mode="json") if facts.price else None,
            "обеспечение": (
                facts.contract_security.model_dump(mode="json") if facts.contract_security else None
            ),
            "срок_исполнения": facts.duration_text,
            "окончание_подачи_заявок": (
                facts.application_deadline.model_dump(mode="json")
                if facts.application_deadline
                else None
            ),
            "этапы": [stage.model_dump(mode="json") for stage in facts.stages],
            "требования": [item.model_dump(mode="json") for item in facts.requirements],
            "ответственность": [item.model_dump(mode="json") for item in facts.penalties],
        }

        try:
            result = await self._provider.complete_json(
                system=SYNTHESIS_SYSTEM,
                user=build_synthesis_prompt(filename=filename, facts=payload),
                schema_model=DocumentSynthesis,
                max_output_tokens=4_000,
            )
            return DocumentSynthesis.model_validate(result.data), result.usage, 1, []
        except (LLMError, LLMResponseInvalidError, ValidationError) as exc:
            # Факты уже извлечены — терять их из-за неудавшегося пересказа глупо.
            logger.warning("Синтез выжимки не удался, собираю по шаблону: %s", exc)
            return (
                build_fallback_synthesis(facts),
                LLMUsage(),
                0,
                ["Итоговая выжимка собрана по шаблону: запрос на синтез не удался."],
            )
