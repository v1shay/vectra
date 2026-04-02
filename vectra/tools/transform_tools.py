from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import resolve_object, validate_vector3
from .registry import register_tool


@register_tool
class TransformObjectTool(BaseTool):
    name = "object.transform"
    description = "Move, rotate, or scale an existing object that is already present in the scene."
    input_schema = {
        "target": {"type": "string", "required": False},
        "object_name": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
        "delta": {"type": "vector3", "required": False},
        "rotation": {"type": "vector3", "required": False},
        "rotation_euler": {"type": "vector3", "required": False},
        "scale": {"type": "vector3", "required": False},
    }
    output_schema = {
        "object_name": {"type": "string"},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)

        target = params.get("target", params.get("object_name"))
        normalized: dict[str, Any] = {}
        if isinstance(target, str) and target.strip():
            normalized["target"] = target.strip()
        if "location" in params:
            normalized["location"] = validate_vector3(params["location"], "location")
        if "delta" in params:
            normalized["delta"] = validate_vector3(params["delta"], "delta")
        rotation_value = params.get("rotation", params.get("rotation_euler"))
        if rotation_value is not None:
            normalized["rotation"] = validate_vector3(
                rotation_value,
                "rotation",
            )
        if "scale" in params:
            normalized["scale"] = validate_vector3(params["scale"], "scale")

        if len(normalized) == 0:
            raise ToolValidationError(
                "At least one target or transform field must be provided"
            )
        if not any(key in normalized for key in {"location", "delta", "rotation", "scale"}):
            raise ToolValidationError(
                "At least one transform field ('location', 'delta', 'rotation', or 'scale') is required"
            )

        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        validated = self.validate_params(params)
        obj = resolve_object(context, validated.get("target"))
        if obj is None:
            raise ToolExecutionError("No transform target could be resolved")

        if "location" in validated:
            obj.location = validated["location"]
        if "delta" in validated:
            current = [float(component) for component in obj.location[:3]]
            delta = validated["delta"]
            obj.location = (
                current[0] + delta[0],
                current[1] + delta[1],
                current[2] + delta[2],
            )
        if "rotation" in validated:
            obj.rotation_euler = validated["rotation"]
        if "scale" in validated:
            obj.scale = validated["scale"]

        view_layer = getattr(bpy.context, "view_layer", None)
        if view_layer is not None:
            view_layer.update()

        return ToolExecutionResult(
            outputs={"object_name": obj.name, "object_names": [obj.name]},
            message=f"Transformed object '{obj.name}'",
        )
