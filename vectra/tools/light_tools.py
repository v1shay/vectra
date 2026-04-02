from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import ensure_object_mode, look_at_rotation, normalize_optional_string, resolve_object, validate_vector3, vector_to_list
from .registry import register_tool


@register_tool
class CreateLightTool(BaseTool):
    name = "light.create"
    description = "Create a light that improves scene readability or mood."
    input_schema = {
        "type": {"type": "string", "required": False, "enum": ["AREA", "POINT", "SUN", "SPOT"]},
        "location": {"type": "vector3", "required": False},
        "energy": {"type": "number", "required": False},
        "rotation": {"type": "vector3", "required": False},
        "target": {"type": "string", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        normalized = {
            "type": str(params.get("type", "AREA")).strip().upper() or "AREA",
            "energy": float(params.get("energy", 1500.0)),
        }
        if normalized["type"] not in {"AREA", "POINT", "SUN", "SPOT"}:
            raise ToolValidationError("'type' must be one of AREA, POINT, SUN, or SPOT")
        if "location" in params and params["location"] is not None:
            normalized["location"] = validate_vector3(params["location"], "location")
        if "rotation" in params and params["rotation"] is not None:
            normalized["rotation"] = validate_vector3(params["rotation"], "rotation")
        if "target" in params:
            normalized["target"] = normalize_optional_string(params["target"], "target")
        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        location = validated.get("location", (4.0, -4.0, 6.0))
        rotation = validated.get("rotation")
        target = resolve_object(context, validated.get("target"))
        if rotation is None and target is not None:
            rotation = look_at_rotation(list(location), vector_to_list(target.location))
        if rotation is None:
            rotation = (0.9, 0.0, 0.8)
        result = bpy.ops.object.light_add(type=validated["type"], location=location, rotation=rotation)
        if isinstance(result, set) and "FINISHED" not in result:
            raise ToolExecutionError(f"Failed to create a light: {result}")
        light_object = bpy.context.active_object
        if light_object is None:
            raise ToolExecutionError("Light creation did not create an active object")
        if getattr(light_object, "data", None) is not None and hasattr(light_object.data, "energy"):
            light_object.data.energy = validated["energy"]
        return ToolExecutionResult(
            outputs={"object_name": light_object.name, "object_names": [light_object.name]},
            message=f"Created light '{light_object.name}'",
        )


@register_tool
class AdjustLightTool(BaseTool):
    name = "light.adjust"
    description = "Adjust an existing light's location, rotation, or energy."
    input_schema = {
        "target": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
        "rotation": {"type": "vector3", "required": False},
        "energy": {"type": "number", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        light_object = resolve_object(context, params.get("target"))
        if light_object is None or getattr(light_object, "type", "") != "LIGHT":
            raise ToolExecutionError("No light could be resolved for adjustment")

        if params.get("location") is not None:
            light_object.location = validate_vector3(params["location"], "location")
        if params.get("rotation") is not None:
            light_object.rotation_euler = validate_vector3(params["rotation"], "rotation")
        if params.get("energy") is not None:
            if isinstance(params["energy"], bool):
                raise ToolValidationError("'energy' must be numeric")
            light_object.data.energy = float(params["energy"])

        return ToolExecutionResult(
            outputs={"object_name": light_object.name, "object_names": [light_object.name]},
            message=f"Adjusted light '{light_object.name}'",
        )
