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
class ObjectKeyframeTool(BaseTool):
    name = "object.keyframe"
    description = "Insert a keyframe for an object transform property."
    input_schema = {
        "target": {"type": "string", "required": False},
        "frame": {"type": "integer", "required": False},
        "property": {"type": "string", "required": False},
        "value": {"type": "vector3", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        obj = resolve_object(context, params.get("target"))
        if obj is None:
            raise ToolExecutionError("No keyframe target could be resolved")

        property_name = str(params.get("property", "location")).strip().lower() or "location"
        if property_name not in {"location", "rotation", "scale"}:
            raise ToolValidationError("'property' must be one of location, rotation, or scale")
        frame = params.get("frame", getattr(context.scene, "frame_current", 1))
        if isinstance(frame, bool):
            raise ToolValidationError("'frame' must be an integer")
        if params.get("value") is not None:
            value = validate_vector3(params["value"], "value")
            if property_name == "rotation":
                obj.rotation_euler = value
            elif property_name == "scale":
                obj.scale = value
            else:
                obj.location = value
        data_path = "rotation_euler" if property_name == "rotation" else property_name
        obj.keyframe_insert(data_path=data_path, frame=int(frame))
        return ToolExecutionResult(
            outputs={"object_name": obj.name, "object_names": [obj.name], "frame": int(frame)},
            message=f"Inserted a {property_name} keyframe for '{obj.name}' at frame {int(frame)}",
        )
