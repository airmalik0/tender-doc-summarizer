"""Тесты генератора строгой JSON Schema.

Ограничения провайдеров проверяются здесь, а не в проде на живом запросе:
именно так был пойман отказ Anthropic на 35 union-типах при лимите 16.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.models.extraction import ChunkExtraction, DocumentSynthesis
from app.services.llm.schema import to_strict_json_schema


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_no_refs_left() -> None:
    dumped = json.dumps(to_strict_json_schema(ChunkExtraction))
    assert "$ref" not in dumped
    assert "$defs" not in dumped


def test_no_union_types() -> None:
    """Лимит Anthropic — 16 параметров с union-типами на схему."""
    dumped = json.dumps(to_strict_json_schema(ChunkExtraction))
    assert "anyOf" not in dumped
    assert '"type": [' not in dumped


def test_every_object_is_strict() -> None:
    schema = to_strict_json_schema(ChunkExtraction)
    for node in _walk(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node.get("properties", {}))


def test_defaults_are_stripped() -> None:
    class WithDefault(BaseModel):
        name: str = Field(default="значение по умолчанию")

    assert "default" not in json.dumps(to_strict_json_schema(WithDefault))


def test_nested_models_are_inlined() -> None:
    schema = to_strict_json_schema(ChunkExtraction)
    price = schema["properties"]["price"]
    assert price["type"] == "object"
    assert "amount" in price["properties"]
    assert price["properties"]["evidence"]["properties"]["page"]["type"] == "integer"


def test_descriptions_survive() -> None:
    """Описания полей — это половина качества извлечения, их терять нельзя."""
    schema = to_strict_json_schema(ChunkExtraction)
    assert schema["properties"]["price"]["description"]
    assert schema["properties"]["price"]["properties"]["amount"]["description"]


def test_synthesis_schema_is_flat() -> None:
    schema = to_strict_json_schema(DocumentSynthesis)
    assert set(schema["properties"]) == {"summary", "risks"}
    assert schema["properties"]["risks"]["type"] == "array"
