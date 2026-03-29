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

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ToolValidationError("Tool params must be a dictionary")
        return params

    @abstractmethod
    def execute(self, context: "bpy.types.Context | Any", params: dict[str, Any]) -> ToolExecutionResult:
        raise NotImplementedError
