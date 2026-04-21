from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import bpy


class ToolValidationError(Exception):
    """Raised when tool params fail validation."""

    def __init__(
        self,
        message: str,
        *,
        missing_params: list[str] | None = None,
        invalid_params: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.missing_params = list(missing_params or [])
        self.invalid_params = list(invalid_params or [])
        self.details = dict(details or {})


class ToolExecutionError(Exception):
    """Raised when a tool cannot complete execution."""


@dataclass
class ToolExecutionResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ToolValidationError(
                "Tool params must be a dictionary",
                invalid_params=["params"],
                details={"received_type": type(params).__name__},
            )

        allowed_keys = set(self.input_schema)
        unknown_keys = sorted(set(params) - allowed_keys)
        if unknown_keys:
            raise ToolValidationError(
                f"Unknown param(s) for '{self.name}': {unknown_keys}",
                invalid_params=unknown_keys,
                details={"allowed_params": sorted(allowed_keys)},
            )

        missing_required = sorted(
            key
            for key, spec in self.input_schema.items()
            if isinstance(spec, dict) and spec.get("required") and key not in params
        )
        if missing_required:
            raise ToolValidationError(
                f"Missing required param(s) for '{self.name}': {missing_required}",
                missing_params=missing_required,
                details={"required_params": missing_required},
            )

        invalid_params: list[str] = []
        invalid_details: dict[str, Any] = {}
        for key, value in params.items():
            spec = self.input_schema.get(key)
            if not isinstance(spec, dict):
                continue
            required = bool(spec.get("required"))
            if value is None:
                if required:
                    invalid_params.append(key)
                    invalid_details[key] = {
                        "expected_type": spec.get("type", "unknown"),
                        "received_type": "NoneType",
                    }
                continue
            type_error = self._schema_type_error(key, value, spec)
            if type_error is not None:
                invalid_params.append(key)
                invalid_details[key] = {
                    "expected_type": spec.get("type", "unknown"),
                    "received_type": type(value).__name__,
                    "message": type_error,
                }
                continue
            enum_values = spec.get("enum")
            if isinstance(enum_values, list) and value not in enum_values:
                invalid_params.append(key)
                invalid_details[key] = {
                    "allowed_values": list(enum_values),
                    "received_value": value,
                    "message": f"'{key}' must be one of {enum_values}",
                }

        if invalid_params:
            single_detail = invalid_details.get(invalid_params[0], {}) if len(invalid_params) == 1 else {}
            message = str(single_detail.get("message", "")).strip() or f"Invalid param(s) for '{self.name}': {sorted(invalid_params)}"
            raise ToolValidationError(
                message,
                invalid_params=sorted(invalid_params),
                details={"invalid_params": invalid_details},
            )

        return dict(params)

    @staticmethod
    def _schema_type_error(field_name: str, value: Any, spec: dict[str, Any]) -> str | None:
        expected_type = str(spec.get("type", "")).strip()
        if not expected_type:
            return None
        if expected_type == "string":
            if not isinstance(value, str):
                return f"'{field_name}' must be a string"
            return None
        if expected_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"'{field_name}' must be numeric"
            if not math.isfinite(float(value)):
                return f"'{field_name}' must be finite"
            return None
        if expected_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return f"'{field_name}' must be an integer"
            return None
        if expected_type == "boolean":
            if not isinstance(value, bool):
                return f"'{field_name}' must be a boolean"
            return None
        if expected_type == "vector3":
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return f"'{field_name}' must be a 3-item list or tuple"
            for component in value:
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    return f"'{field_name}' values must be numeric"
                if not math.isfinite(float(component)):
                    return f"'{field_name}' values must be finite"
            return None
        if expected_type == "string_array":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                return f"'{field_name}' must be a list of strings"
            return None
        return None

    @abstractmethod
    def execute(self, context: "bpy.types.Context | Any", params: dict[str, Any]) -> ToolExecutionResult:
        raise NotImplementedError
