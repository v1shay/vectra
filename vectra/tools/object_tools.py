from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import (
    default_spacing_for_object,
    ensure_object_mode,
    normalize_optional_string,
    resolve_object,
    resolve_objects,
    validate_vector3,
)
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
class DeleteManyObjectsTool(BaseTool):
    name = "object.delete_many"
    description = "Delete several objects at once."
    input_schema = {
        "targets": {"type": "string_array", "required": False},
        "objects": {"type": "string_array", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        targets = params.get("targets", params.get("objects"))
        resolved = resolve_objects(context, targets)
        if not resolved:
            raise ToolExecutionError("No deletion targets could be resolved")
        deleted = [obj.name for obj in resolved]
        for obj in resolved:
            bpy.data.objects.remove(obj, do_unlink=True)
        return ToolExecutionResult(
            outputs={"object_name": deleted[0], "object_names": deleted},
            message=f"Deleted {len(deleted)} object(s)",
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


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _dimension_for_axis(obj: Any, axis: str) -> float:
    dimensions = getattr(obj, "dimensions", (2.0, 2.0, 2.0))
    index = _axis_index(axis)
    try:
        return max(float(dimensions[index]), 0.5)
    except (TypeError, ValueError, IndexError):
        return 1.0


@register_tool
class ParentObjectsTool(BaseTool):
    name = "object.parent"
    description = "Parent one or more child objects under a parent object."
    input_schema = {
        "parent": {"type": "string", "required": False},
        "children": {"type": "string_array", "required": False},
        "objects": {"type": "string_array", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        parent = resolve_object(context, params.get("parent"))
        if parent is None:
            raise ToolExecutionError("No parent object could be resolved")
        children = resolve_objects(context, params.get("children", params.get("objects")))
        if not children:
            raise ToolExecutionError("No child objects could be resolved")
        parented: list[str] = []
        for child in children:
            if child == parent:
                continue
            child.parent = parent
            parented.append(child.name)
        if not parented:
            raise ToolExecutionError("No valid child objects remained after parent resolution")
        return ToolExecutionResult(
            outputs={"object_name": parent.name, "object_names": parented},
            message=f"Parented {len(parented)} object(s) to '{parent.name}'",
        )


@register_tool
class DistributeObjectsTool(BaseTool):
    name = "object.distribute"
    description = "Distribute several objects along one axis using their dimensions and a gap."
    input_schema = {
        "targets": {"type": "string_array", "required": False},
        "objects": {"type": "string_array", "required": False},
        "axis": {"type": "string", "required": False},
        "gap": {"type": "number", "required": False},
        "anchor": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        axis = str(params.get("axis", "x")).strip().lower() or "x"
        if axis not in {"x", "y", "z"}:
            raise ToolValidationError("'axis' must be one of x, y, or z")
        gap = params.get("gap", 0.5)
        if isinstance(gap, bool):
            raise ToolValidationError("'gap' must be numeric")
        normalized = {
            "axis": axis,
            "gap": float(gap),
            "targets": params.get("targets", params.get("objects")),
        }
        if params.get("anchor") is not None:
            normalized["anchor"] = validate_vector3(params["anchor"], "anchor")
        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        objects = resolve_objects(context, validated.get("targets"))
        if len(objects) < 2:
            raise ToolExecutionError("At least two objects are required to distribute them")

        axis = validated["axis"]
        index = _axis_index(axis)
        objects = sorted(objects, key=lambda obj: float(obj.location[index]))
        anchor = list(validated.get("anchor") or [float(objects[0].location[0]), float(objects[0].location[1]), float(objects[0].location[2])])

        cursor = float(anchor[index])
        touched: list[str] = []
        for position, obj in enumerate(objects):
            location = [float(component) for component in obj.location[:3]]
            if position == 0:
                location[index] = cursor
            else:
                prev_width = _dimension_for_axis(objects[position - 1], axis) / 2.0
                current_width = _dimension_for_axis(obj, axis) / 2.0
                cursor += prev_width + current_width + float(validated["gap"])
                location[index] = cursor
            obj.location = tuple(location)
            touched.append(obj.name)

        return ToolExecutionResult(
            outputs={"object_name": touched[0], "object_names": touched},
            message=f"Distributed {len(touched)} object(s) along the {axis.upper()} axis",
        )


@register_tool
class AlignObjectsTool(BaseTool):
    name = "object.align"
    description = "Align several objects along a chosen axis to a target or shared value."
    input_schema = {
        "targets": {"type": "string_array", "required": False},
        "objects": {"type": "string_array", "required": False},
        "target": {"type": "string", "required": False},
        "axis": {"type": "string", "required": False},
        "value": {"type": "number", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        axis = str(params.get("axis", "z")).strip().lower() or "z"
        if axis not in {"x", "y", "z"}:
            raise ToolValidationError("'axis' must be one of x, y, or z")
        normalized: dict[str, Any] = {
            "axis": axis,
            "targets": params.get("targets", params.get("objects")),
            "target": params.get("target"),
        }
        if params.get("value") is not None:
            if isinstance(params["value"], bool):
                raise ToolValidationError("'value' must be numeric")
            normalized["value"] = float(params["value"])
        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        objects = resolve_objects(context, validated.get("targets"))
        if not objects:
            raise ToolExecutionError("No alignment targets could be resolved")
        axis = validated["axis"]
        index = _axis_index(axis)

        anchor_value = validated.get("value")
        if anchor_value is None:
            target = resolve_object(context, validated.get("target"))
            if target is not None:
                anchor_value = float(target.location[index])
            else:
                anchor_value = sum(float(obj.location[index]) for obj in objects) / float(len(objects))

        touched: list[str] = []
        for obj in objects:
            location = [float(component) for component in obj.location[:3]]
            location[index] = float(anchor_value)
            obj.location = tuple(location)
            touched.append(obj.name)

        return ToolExecutionResult(
            outputs={"object_name": touched[0], "object_names": touched},
            message=f"Aligned {len(touched)} object(s) on the {axis.upper()} axis",
        )
