from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult
from .helpers import look_at_rotation, resolve_object, validate_vector3, vector_to_list
from .registry import register_tool


@register_tool
class EnsureCameraTool(BaseTool):
    name = "camera.ensure"
    description = "Ensure the scene has a useful camera and optionally aim it at a target."
    input_schema = {
        "location": {"type": "vector3", "required": False},
        "rotation": {"type": "vector3", "required": False},
        "target": {"type": "string", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        existing_camera = getattr(context.scene, "camera", None)
        camera_object = existing_camera
        if camera_object is None:
            result = bpy.ops.object.camera_add(location=(8.0, -8.0, 6.0))
            if isinstance(result, set) and "FINISHED" not in result:
                raise ToolExecutionError(f"Failed to create a camera: {result}")
            camera_object = bpy.context.active_object
            context.scene.camera = camera_object

        if camera_object is None:
            raise ToolExecutionError("No camera could be resolved")

        if "location" in params and params["location"] is not None:
            camera_object.location = validate_vector3(params["location"], "location")
        target = resolve_object(context, params.get("target"))
        if "rotation" in params and params["rotation"] is not None:
            camera_object.rotation_euler = validate_vector3(params["rotation"], "rotation")
        elif target is not None:
            camera_object.rotation_euler = look_at_rotation(
                vector_to_list(camera_object.location),
                vector_to_list(target.location),
            )

        return ToolExecutionResult(
            outputs={"object_name": camera_object.name, "object_names": [camera_object.name]},
            message=f"Ensured camera '{camera_object.name}'",
        )
