"""Выбор провайдера по конфигурации.

Правило простое и предсказуемое: явно названный провайдер обязан
подняться — если ключа нет, это ошибка конфигурации, а не повод молча
понизить режим. Режим auto, наоборот, берёт первого доступного и
скатывается в offline, честно сообщая об этом в warnings ответа.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings
from app.core.exceptions import ProviderNotConfiguredError
from app.core.logging import get_logger
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import LLMProvider
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.offline import OfflineProvider
from app.services.llm.openai_provider import OpenAIProvider

logger = get_logger(__name__)

# Порядок приоритета в режиме auto.
PRIORITY = ("anthropic", "openai", "gemini")

# Тип указан явно: без него mypy сводит значения словаря к type[LLMProvider]
# и справедливо ругается на создание абстрактного класса.
_CONSTRUCTORS: dict[str, Callable[[Settings], LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def resolve_provider_name(settings: Settings) -> str:
    """Какой провайдер будет использован — без создания клиента.

    Нужно эндпоинту /health, чтобы отвечать быстро и не держать соединений.
    """
    if settings.llm_provider != "auto":
        return settings.llm_provider

    configured = settings.configured_providers()
    for name in PRIORITY:
        if name in configured:
            return name
    return "offline"


def build_provider(settings: Settings) -> LLMProvider:
    """Создаёт провайдера согласно конфигурации."""
    name = resolve_provider_name(settings)

    if name == "offline":
        if settings.llm_provider == "auto":
            logger.warning(
                "Ни один API-ключ не задан — работаю на детерминированных правилах. "
                "Задайте ANTHROPIC_API_KEY в .env, чтобы включить полноценное извлечение."
            )
        return OfflineProvider()

    constructor = _CONSTRUCTORS.get(name)
    if constructor is None:  # pragma: no cover — защищено валидацией настроек
        raise ProviderNotConfiguredError(f"Неизвестный провайдер: {name}")

    try:
        provider = constructor(settings)
    except ValueError as exc:
        raise ProviderNotConfiguredError(
            f"Провайдер «{name}» выбран явно, но не настроен: {exc}. "
            f"Задайте ключ в .env либо переключите LLM_PROVIDER на auto или offline.",
            details={"provider": name},
        ) from exc
    except ImportError as exc:  # pragma: no cover — зависит от сборки окружения
        raise ProviderNotConfiguredError(
            f"SDK провайдера «{name}» не установлен: {exc}", details={"provider": name}
        ) from exc

    logger.info("Провайдер: %s, модель: %s", provider.name, provider.model)
    return provider
