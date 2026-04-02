from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import default_spacing_for_object, ensure_object_mode, normalize_optional_string, resolve_object, resolve_objects, validate_vector3
from .registry import register_tool


@register_tool
class DuplicateObjectTool(BaseTool):
    name = "object.duplicate"
    description = "Duplicate an existing object, optionally creating several copies with an offset."
    input_schema = {
        "target": {"type": "string", "required": False},
        "count": {"type": "integer", "required": False},
        "offset": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        normalized: dict[str, Any] = {}
        if "target" in params:
            normalized["target"] = normalize_optional_string(params["target"], "target")
        count = params.get("count", 1)
        if isinstance(count, bool):
            raise ToolValidationError("'count' must be an integer")
        if count is None:
            count = 1
        count = int(count)
        if count < 1:
            raise ToolValidationError("'count' must be at least 1")
        normalized["count"] = count
        if "offset" in params and params["offset"] is not None:
            normalized["offset"] = validate_vector3(params["offset"], "offset")
        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        source = resolve_object(context, validated.get("target"))
        if source is None:
            raise ToolExecutionError("No duplication target could be resolved")

        offset = validated.get("offset")
        if offset is None:
            spacing = default_spacing_for_object(source)
            offset = (spacing, 0.0, 0.0)

        created_names: list[str] = []
        scene_collection = context.scene.collection
        for index in range(validated["count"]):
            duplicate = source.copy()
            duplicate.data = getattr(source.data, "copy", lambda: source.data)()
            duplicate.location = (
                float(source.location[0]) + offset[0] * float(index + 1),
                float(source.location[1]) + offset[1] * float(index + 1),
                float(source.location[2]) + offset[2] * float(index + 1),
            )
            scene_collection.objects.link(duplicate)
            created_names.append(duplicate.name)

        return ToolExecutionResult(
            outputs={"object_name": created_names[0], "object_names": created_names},
            message=f"Duplicated '{source.name}' {validated['count']} time(s)",
        )


@register_tool
class DeleteObjectTool(BaseTool):
    name = "object.delete"
    description = "Delete one resolved object from the scene."
    input_schema = {
        "target": {"type": "string", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        target = resolve_object(context, params.get("target"))
        if target is None:
            raise ToolExecutionError("No deletion target could be resolved")
        object_name = target.name
        bpy.data.objects.remove(target, do_unlink=True)
        return ToolExecutionResult(
            outputs={"object_name": object_name, "object_names": [object_name]},
            message=f"Deleted object '{object_name}'",
        )


@register_tool
class SelectObjectTool(BaseTool):
    name = "object.select"
    description = "Select one resolved object and optionally clear the current selection first."
    input_schema = {
        "target": {"type": "string", "required": False},
        "mode": {"type": "string", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        mode = str(params.get("mode", "replace")).strip().lower()
        target = resolve_object(context, params.get("target"))
        if target is None:
            raise ToolExecutionError("No selection target could be resolved")
        if mode in {"replace", "set", ""}:
            bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target
        return ToolExecutionResult(
            outputs={"object_name": target.name, "object_names": [target.name]},
            message=f"Selected object '{target.name}'",
        )
