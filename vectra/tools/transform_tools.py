from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .mesh_tools import _validate_vector3
from .registry import register_tool


@register_tool
class TransformObjectTool(BaseTool):
    name = "object.transform"
    description = "Apply deterministic object transforms"
    input_schema = {
        "object_name": {"type": "string", "required": True},
        "location": {"type": "vector3", "required": False},
        "rotation_euler": {"type": "vector3", "required": False},
        "scale": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)

        object_name = params.get("object_name")
        if not isinstance(object_name, str) or not object_name:
            raise ToolValidationError("'object_name' must be a non-empty string")

        normalized: dict[str, Any] = {"object_name": object_name}
        if "location" in params:
            normalized["location"] = _validate_vector3(params["location"], "location")
        if "rotation_euler" in params:
            normalized["rotation_euler"] = _validate_vector3(
                params["rotation_euler"],
                "rotation_euler",
            )
        if "scale" in params:
            normalized["scale"] = _validate_vector3(params["scale"], "scale")

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
            obj.location = validated["location"]
        if "rotation_euler" in validated:
            obj.rotation_euler = validated["rotation_euler"]
        if "scale" in validated:
            obj.scale = validated["scale"]

        view_layer = getattr(bpy.context, "view_layer", None)
        if view_layer is not None:
            view_layer.update()

        return ToolExecutionResult(
            outputs={"object_name": obj.name},
            message=f"Transformed object '{obj.name}'",
        )
