from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import look_at_rotation, resolve_object, validate_vector3, vector_to_list
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


@register_tool
class CameraOrbitAnimationTool(BaseTool):
    name = "animation.camera_orbit"
    description = "Create a short camera move around the scene or a focal target."
    input_schema = {
        "target": {"type": "string", "required": False},
        "start_frame": {"type": "integer", "required": False},
        "end_frame": {"type": "integer", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        target = resolve_object(context, params.get("target"))
        target_location = vector_to_list(target.location) if target is not None else [0.0, 0.0, 0.8]
        camera = getattr(context.scene, "camera", None)
        if camera is None or getattr(camera, "type", "") != "CAMERA":
            result = bpy.ops.object.camera_add(location=(5.2, -6.0, 3.0))
            if isinstance(result, set) and "FINISHED" not in result:
                raise ToolExecutionError(f"Failed to create camera for animation: {result}")
            camera = bpy.context.active_object
            context.scene.camera = camera
        start_frame = int(params.get("start_frame", 1))
        end_frame = int(params.get("end_frame", 72))
        if end_frame <= start_frame:
            raise ToolValidationError("'end_frame' must be greater than 'start_frame'")
        start_location = [target_location[0] + 5.2, target_location[1] - 5.4, target_location[2] + 2.7]
        end_location = [target_location[0] - 4.6, target_location[1] - 5.0, target_location[2] + 3.1]
        context.scene.frame_set(start_frame)
        camera.location = start_location
        camera.rotation_euler = look_at_rotation(start_location, target_location)
        camera.keyframe_insert(data_path="location", frame=start_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=start_frame)
        context.scene.frame_set(end_frame)
        camera.location = end_location
        camera.rotation_euler = look_at_rotation(end_location, target_location)
        camera.keyframe_insert(data_path="location", frame=end_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=end_frame)
        return ToolExecutionResult(
            outputs={"object_name": camera.name, "object_names": [camera.name], "frame_start": start_frame, "frame_end": end_frame},
            message=f"Animated camera '{camera.name}' from frame {start_frame} to {end_frame}",
        )
