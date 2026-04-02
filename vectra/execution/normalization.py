from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..tools.base import BaseTool


@dataclass(frozen=True)
class ParamRepair:
    field: str
    reason: str


@dataclass(frozen=True)
class NormalizedParams:
    params: dict[str, Any]
    repairs: list[ParamRepair] = field(default_factory=list)


def _coerce_scalar(value: Any, expected_type: str) -> tuple[Any, list[ParamRepair]]:
    repairs: list[ParamRepair] = []
    if expected_type == "integer" and isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            repairs.append(ParamRepair(field="", reason=f"Coerced integer from string '{value}'"))
            return int(stripped), repairs
    if expected_type == "number" and isinstance(value, str):
        stripped = value.strip()
        try:
            number = float(stripped)
        except ValueError:
            return value, repairs
        repairs.append(ParamRepair(field="", reason=f"Coerced number from string '{value}'"))
        return number, repairs
    if expected_type == "string_array":
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                repairs.append(ParamRepair(field="", reason="Dropped empty string array value"))
                return [], repairs
            if "," in stripped:
                values = [item.strip() for item in stripped.split(",") if item.strip()]
            else:
                values = [stripped]
            repairs.append(ParamRepair(field="", reason="Coerced string input into a string array"))
            return values, repairs
    return value, repairs


def _drop_absent_optional_fields(tool: BaseTool, params: dict[str, Any]) -> tuple[dict[str, Any], list[ParamRepair]]:
    normalized: dict[str, Any] = {}
    repairs: list[ParamRepair] = []
    for key, value in params.items():
        spec = tool.input_schema.get(key)
        required = isinstance(spec, dict) and bool(spec.get("required"))
        if value is None and not required:
            repairs.append(ParamRepair(field=key, reason=f"Dropped absent optional field '{key}'"))
            continue
        normalized[key] = value
    return normalized, repairs


def _apply_alias_repairs(tool: BaseTool, params: dict[str, Any]) -> tuple[dict[str, Any], list[ParamRepair]]:
    repairs: list[ParamRepair] = []
    normalized = dict(params)

    alias_pairs = {
        "mesh.create_primitive": {"primitive_type": "type"},
        "object.transform": {"object_name": "target", "rotation_euler": "rotation"},
        "object.transform_many": {"objects": "targets", "rotation_euler": "rotation"},
        "object.delete_many": {"objects": "targets"},
        "object.distribute": {"objects": "targets"},
        "object.align": {"objects": "targets"},
        "object.parent": {"objects": "children"},
    }.get(tool.name, {})

    for source, target in alias_pairs.items():
        if target not in tool.input_schema:
            continue
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
            repairs.append(ParamRepair(field=target, reason=f"Repaired alias '{source}' to '{target}'"))

    return normalized, repairs


def normalize_action_params(tool: BaseTool, params: Mapping[str, Any]) -> NormalizedParams:
    normalized, repairs = _apply_alias_repairs(tool, dict(params))
    normalized, dropped_repairs = _drop_absent_optional_fields(tool, normalized)
    repairs.extend(dropped_repairs)

    for key, spec in tool.input_schema.items():
        if key not in normalized or not isinstance(spec, dict):
            continue
        expected_type = str(spec.get("type", "")).strip()
        value, field_repairs = _coerce_scalar(normalized[key], expected_type)
        if field_repairs:
            normalized[key] = value
            repairs.extend(
                ParamRepair(field=key, reason=repair.reason.replace("field ''", key))
                for repair in field_repairs
            )

    return NormalizedParams(params=normalized, repairs=repairs)
