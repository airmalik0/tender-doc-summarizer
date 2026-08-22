"""OCR-обвязка над tesseract.

Отдельный модуль по двум причинам: tesseract — внешний бинарник, которого
может не быть в системе, и его отсутствие не должно ронять импорт пакета;
плюс так OCR легко подменить в тестах.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    """Установлен ли tesseract в системе.

    Результат кэшируется: проверка дёргает внешний процесс, а ответ за время
    жизни контейнера не меняется.
    """
    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
    except Exception as exc:  # pragma: no cover — зависит от окружения
        logger.warning("OCR недоступен: %s", exc)
        return False

    logger.info("OCR доступен, tesseract %s", version)
    return True


@lru_cache(maxsize=1)
def available_languages() -> frozenset[str]:
    """Языковые пакеты, установленные для tesseract."""
    if not ocr_available():
        return frozenset()
    try:
        import pytesseract

        return frozenset(pytesseract.get_languages(config=""))
    except Exception:  # pragma: no cover — зависит от окружения
        return frozenset()


def resolve_lang(requested: str) -> str:
    """Отбрасывает языки, которых нет в системе.

    Просьба распознать «rus+eng» на образе без русского пакета уронила бы
    tesseract целиком; вместо этого молча распознаём тем, что есть.
    """
    installed = available_languages()
    if not installed:
        return requested
    wanted = [code for code in requested.split("+") if code]
    usable = [code for code in wanted if code in installed]
    if not usable:
        logger.warning("Ни один из языков %s не установлен, откатываюсь на eng", "+".join(wanted))
        return "eng" if "eng" in installed else next(iter(sorted(installed)))
    if len(usable) != len(wanted):
        missing = set(wanted) - set(usable)
        logger.warning("Языковые пакеты tesseract не найдены: %s", ", ".join(sorted(missing)))
    return "+".join(usable)


def recognize(image: Image.Image, lang: str) -> str:
    """Распознаёт текст на изображении страницы."""
    import pytesseract

    # PSM 6 — «единый блок текста»: для страницы документа заметно устойчивее
    # автоматической сегментации, которая на таблицах любит терять строки.
    return pytesseract.image_to_string(image, lang=lang, config="--psm 6")
