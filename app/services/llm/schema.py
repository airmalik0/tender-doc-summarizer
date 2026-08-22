"""Превращение Pydantic-модели в строгую JSON Schema для провайдеров.

Режимы структурированного вывода у Anthropic, OpenAI и Google принимают
не любую JSON Schema, а её подмножество, и требования у них похожие:
у каждого объекта должны быть перечислены все свойства в required и
запрещены дополнительные. Pydantic же по умолчанию выносит вложенные
модели в $defs и делает необязательные поля отсутствующими в required.

Здесь схема приводится к общему знаменателю один раз, а не переписывается
руками под каждый SDK — иначе схема и модель разъедутся при первой же
правке полей.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

# Ограничение на глубину РАЗВОРОТА ССЫЛОК, а не на вложенность схемы:
# рекурсивная модель (узел, ссылающийся на себя) иначе развернулась бы бесконечно.
_MAX_REF_DEPTH = 8


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Строгая, полностью развёрнутая схема модели."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    inlined = _inline_refs(schema, definitions, ref_depth=0)
    return _tighten(inlined)


def _inline_refs(node: Any, definitions: dict[str, Any], ref_depth: int) -> Any:
    """Разворачивает $ref прямо в схему.

    Ссылки поддерживаются не всеми провайдерами (у Google исторически с ними
    беда), поэтому надёжнее отдать самодостаточное дерево.
    """
    if ref_depth > _MAX_REF_DEPTH:
        raise ValueError("Схема содержит рекурсивную ссылку — развернуть её нельзя")

    if isinstance(node, dict):
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            if name not in definitions:
                raise ValueError(f"Не найдено определение для ссылки {node['$ref']}")
            resolved = _inline_refs(copy.deepcopy(definitions[name]), definitions, ref_depth + 1)
            # Ключи рядом со ссылкой (description и подобные) должны пережить разворот.
            extra = {key: value for key, value in node.items() if key != "$ref"}
            resolved.update(_inline_refs(extra, definitions, ref_depth))
            return resolved
        return {key: _inline_refs(value, definitions, ref_depth) for key, value in node.items()}

    if isinstance(node, list):
        return [_inline_refs(item, definitions, ref_depth) for item in node]

    return node


def _tighten(node: Any) -> Any:
    """Делает все свойства обязательными и запрещает лишние ключи."""
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {key: _tighten(value) for key, value in node.items() if key != "default"}

    if result.get("type") == "object" or "properties" in result:
        properties = result.get("properties", {})
        result["additionalProperties"] = False
        # Необязательные поля уже описаны как «тип или null», поэтому требование
        # присутствия ключа ничего не ломает: модель обязана явно вернуть null,
        # а не молча пропустить поле.
        result["required"] = list(properties.keys())

    return result
