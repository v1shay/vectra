from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .mesh_tools import _validate_vector3
from .registry import register_tool


def _normalize_transform_value(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if not value:
            raise ToolValidationError(f"'{field_name}' mapping must include at least one axis")

        unexpected_axes = sorted(set(value) - {"x", "y", "z"})
        if unexpected_axes:
            raise ToolValidationError(
                f"'{field_name}' mapping may only use x, y, and z axes"
            )

        normalized: dict[str, float] = {}
        for axis, raw_component in value.items():
            try:
                normalized[axis] = float(raw_component)
            except (TypeError, ValueError) as exc:
                raise ToolValidationError(f"'{field_name}' values must be numeric") from exc
        return normalized

    return _validate_vector3(value, field_name)


@register_tool
class TransformObjectTool(BaseTool):
    name = "object.transform"
    description = "Move, rotate, or scale an existing object that is already present in the scene."
    input_schema = {
        "object_name": {"type": "string", "required": True},
        "location": {"type": "vector3", "required": False},
        "rotation_euler": {"type": "vector3", "required": False},
        "scale": {"type": "vector3", "required": False},
    }
    output_schema = {
        "object_name": {"type": "string"},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)

        object_name = params.get("object_name")
        if not isinstance(object_name, str) or not object_name:
            raise ToolValidationError("'object_name' must be a non-empty string")

        normalized: dict[str, Any] = {"object_name": object_name}
        if "location" in params and params["location"] is not None:
            normalized["location"] = _normalize_transform_value(params["location"], "location")
        if "rotation_euler" in params and params["rotation_euler"] is not None:
            normalized["rotation_euler"] = _normalize_transform_value(
                params["rotation_euler"],
                "rotation_euler",
            )
        if "scale" in params and params["scale"] is not None:
            normalized["scale"] = _normalize_transform_value(params["scale"], "scale")

        if len(normalized) == 1:
            raise ToolValidationError(
                "At least one transform field ('location', 'rotation_euler', or 'scale') is required"
            )

        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        del context
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        validated = self.validate_params(params)
        obj = bpy.data.objects.get(validated["object_name"])
        if obj is None:
            raise ToolExecutionError(f"Object '{validated['object_name']}' was not found")

        if "location" in validated:
            obj.location = _validate_vector3(
                validated["location"],
                "location",
                default=tuple(float(component) for component in obj.location[:3]),
            )
        if "rotation_euler" in validated:
            obj.rotation_euler = _validate_vector3(
                validated["rotation_euler"],
                "rotation_euler",
                default=tuple(float(component) for component in obj.rotation_euler[:3]),
            )
        if "scale" in validated:
            obj.scale = _validate_vector3(
                validated["scale"],
                "scale",
                default=tuple(float(component) for component in obj.scale[:3]),
            )

        view_layer = getattr(bpy.context, "view_layer", None)
        if view_layer is not None:
            view_layer.update()

        return ToolExecutionResult(
            outputs={"object_name": obj.name},
            message=f"Transformed object '{obj.name}'",
        )
