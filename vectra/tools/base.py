from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import bpy


class ToolValidationError(Exception):
    """Raised when tool params fail validation."""


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
            raise ToolValidationError("Tool params must be a dictionary")

        allowed_keys = set(self.input_schema)
        unknown_keys = sorted(set(params) - allowed_keys)
        if unknown_keys:
            raise ToolValidationError(
                f"Unknown param(s) for '{self.name}': {unknown_keys}"
            )

        missing_required = sorted(
            key
            for key, spec in self.input_schema.items()
            if isinstance(spec, dict) and spec.get("required") and key not in params
        )
        if missing_required:
            raise ToolValidationError(
                f"Missing required param(s) for '{self.name}': {missing_required}"
            )

        return dict(params)

    @abstractmethod
    def execute(self, context: "bpy.types.Context | Any", params: dict[str, Any]) -> ToolExecutionResult:
        raise NotImplementedError
