"""Зависимости FastAPI.

Провайдер, кэш и пайплайн создаются один раз на старте приложения и живут
в app.state: клиент провайдера держит пул соединений, и пересоздавать его
на каждый запрос — верный способ упереться в лимиты.
"""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.services.extraction.pipeline import SummarizationPipeline
from app.services.llm.base import LLMProvider


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_provider(request: Request) -> LLMProvider:
    return request.app.state.provider


def get_pipeline(request: Request) -> SummarizationPipeline:
    return request.app.state.pipeline
