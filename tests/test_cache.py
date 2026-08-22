"""Тесты кэша разборов."""

from __future__ import annotations

from pathlib import Path

from app.models.summary import DocumentInfo, ProcessingMeta, TenderSummary
from app.services.cache import SummaryCache, content_hash


def _summary() -> TenderSummary:
    return TenderSummary(
        document=DocumentInfo(
            filename="t.pdf", sha256="a" * 64, pages=1, characters=10, text_source="text_layer"
        ),
        meta=ProcessingMeta(
            provider="scripted", chunks=1, llm_calls=1, pdf_ms=1, llm_ms=1, total_ms=2
        ),
    )


def test_hash_depends_on_content() -> None:
    assert content_hash(b"a") != content_hash(b"b")
    assert content_hash(b"a") == content_hash(b"a")


def test_roundtrip(tmp_path) -> None:
    cache = SummaryCache(tmp_path, enabled=True)
    key = cache.key("a" * 64, "anthropic", "claude-opus-5")

    assert cache.get(key) is None
    cache.put(key, _summary())
    assert cache.get(key) is not None


def test_key_depends_on_model() -> None:
    """Смена модели должна обесценивать прежние записи, а не подсовывать чужой разбор."""
    cache = SummaryCache(Path("."), enabled=False)
    assert cache.key("a" * 64, "anthropic", "opus") != cache.key("a" * 64, "anthropic", "sonnet")
    assert cache.key("a" * 64, "anthropic", "opus") != cache.key("a" * 64, "openai", "opus")


def test_disabled_cache_stores_nothing(tmp_path) -> None:
    cache = SummaryCache(tmp_path, enabled=False)
    key = cache.key("a" * 64, "offline", None)
    cache.put(key, _summary())
    assert cache.get(key) is None


def test_corrupted_entry_is_dropped(tmp_path) -> None:
    cache = SummaryCache(tmp_path, enabled=True)
    key = cache.key("a" * 64, "offline", None)
    (tmp_path / f"{key}.json").write_text("{битый json", encoding="utf-8")

    assert cache.get(key) is None
    assert not (tmp_path / f"{key}.json").exists()
