from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from vectra.agent.observation import build_scene_state, capture_viewport_screenshot

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import active_scene, ensure_object_mode, normalize_optional_string, resolve_objects
from .registry import register_tool


@register_tool
class GroupSceneObjectsTool(BaseTool):
    name = "scene.group"
    description = "Create a collection-style group from provided or inferred objects."
    input_schema = {
        "objects": {"type": "string_array", "required": False},
        "name": {"type": "string", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        normalized: dict[str, Any] = {}
        if "objects" in params:
            objects = params["objects"]
            if not isinstance(objects, list):
                raise ToolValidationError("'objects' must be a list of object names")
            normalized["objects"] = [str(item).strip() for item in objects if str(item).strip()]
        if "name" in params:
            normalized["name"] = normalize_optional_string(params["name"], "name")
        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        objects = resolve_objects(context, validated.get("objects"))
        if not objects:
            raise ToolExecutionError("No objects were available to group")
        collection_name = validated.get("name") or f"VectraGroup_{len(bpy.data.collections) + 1}"
        collection = bpy.data.collections.new(collection_name)
        active_scene(context).collection.children.link(collection)
        existing_names = {candidate.name for candidate in collection.objects}
        for obj in objects:
            if obj.name not in existing_names:
                collection.objects.link(obj)
        return ToolExecutionResult(
            outputs={
                "group_name": collection.name,
                "object_names": [obj.name for obj in objects],
            },
            message=f"Grouped {len(objects)} object(s) into '{collection.name}'",
        )


@register_tool
class GetSceneStateTool(BaseTool):
    name = "scene.get_state"
    description = "Return a compact scene summary for reasoning or verification."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        del params
        return ToolExecutionResult(
            outputs={"scene_state": build_scene_state(context)},
            message="Captured the current scene state",
        )


@register_tool
class CaptureViewTool(BaseTool):
    name = "scene.capture_view"
    description = "Capture a viewport screenshot when visual verification is useful."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        del context, params
        result = capture_viewport_screenshot(iteration=0)
        return ToolExecutionResult(
            outputs={"screenshot": result},
            message="Captured the current viewport" if result.get("available") else str(result.get("reason", "Viewport capture unavailable")),
        )


@register_tool
class FrameViewTool(BaseTool):
    name = "scene.frame_view"
    description = "Frame the current view around the scene or target objects."
    input_schema = {
        "targets": {"type": "string_array", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        del params
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        if bool(getattr(getattr(bpy, "app", None), "background", False)):
            return ToolExecutionResult(outputs={}, message="Frame view skipped in background mode")
        try:
            bpy.ops.view3d.view_all(center=False)
        except RuntimeError:
            return ToolExecutionResult(outputs={}, message="Frame view was unavailable in the current UI context")
        return ToolExecutionResult(outputs={}, message="Framed the current view")


@register_tool
class SetFrameTool(BaseTool):
    name = "scene.set_frame"
    description = "Set the active scene frame for animation work."
    input_schema = {
        "frame": {"type": "integer", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        frame = params.get("frame", 1)
        if isinstance(frame, bool):
            raise ToolValidationError("'frame' must be an integer")
        active_scene(context).frame_set(int(frame))
        return ToolExecutionResult(outputs={"frame": int(frame)}, message=f"Moved to frame {int(frame)}")
