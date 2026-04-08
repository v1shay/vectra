from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import ensure_object_mode, resolve_object
from .registry import register_tool

_SUPPORTED_MODIFIERS = {"BEVEL", "SOLIDIFY", "SUBSURF"}


@register_tool
class AddModifierTool(BaseTool):
    name = "object.add_modifier"
    description = "Add a constrained non-destructive modifier to a mesh object."
    input_schema = {
        "target": {"type": "string", "required": True},
        "modifier_type": {"type": "string", "required": True, "enum": sorted(_SUPPORTED_MODIFIERS)},
        "width": {"type": "number", "required": False},
        "segments": {"type": "integer", "required": False},
        "thickness": {"type": "number", "required": False},
        "levels": {"type": "integer", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = str(params.get("target", "")).strip()
        modifier_type = str(params.get("modifier_type", "")).strip().upper()
        if not target:
            raise ToolValidationError("'target' must be a non-empty object name")
        if modifier_type not in _SUPPORTED_MODIFIERS:
            raise ToolValidationError(f"'modifier_type' must be one of {sorted(_SUPPORTED_MODIFIERS)}")

        normalized: dict[str, Any] = {
            "target": target,
            "modifier_type": modifier_type,
        }
        if params.get("width") is not None:
            if isinstance(params["width"], bool):
                raise ToolValidationError("'width' must be numeric")
            normalized["width"] = float(params["width"])
        if params.get("segments") is not None:
            if isinstance(params["segments"], bool):
                raise ToolValidationError("'segments' must be an integer")
            normalized["segments"] = max(int(params["segments"]), 1)
        if params.get("thickness") is not None:
            if isinstance(params["thickness"], bool):
                raise ToolValidationError("'thickness' must be numeric")
            normalized["thickness"] = float(params["thickness"])
        if params.get("levels") is not None:
            if isinstance(params["levels"], bool):
                raise ToolValidationError("'levels' must be an integer")
            normalized["levels"] = max(int(params["levels"]), 0)
        return normalized

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        ensure_object_mode(context)
        validated = self.validate_params(params)
        obj = resolve_object(context, validated["target"])
        if obj is None:
            raise ToolExecutionError("No modifier target could be resolved")
        if getattr(obj, "type", "") != "MESH":
            raise ToolExecutionError(f"Object '{obj.name}' does not support mesh modifiers")

        modifier_type = validated["modifier_type"]
        modifier = obj.modifiers.new(name=f"Vectra_{modifier_type.title()}", type=modifier_type)
        if modifier_type == "BEVEL":
            modifier.width = float(validated.get("width", 0.05))
            modifier.segments = int(validated.get("segments", 2))
        elif modifier_type == "SOLIDIFY":
            modifier.thickness = float(validated.get("thickness", 0.08))
        elif modifier_type == "SUBSURF":
            levels = int(validated.get("levels", 2))
            modifier.levels = levels
            if hasattr(modifier, "render_levels"):
                modifier.render_levels = levels

        return ToolExecutionResult(
            outputs={
                "object_name": obj.name,
                "object_names": [obj.name],
                "modifier_name": modifier.name,
                "modifier_type": modifier_type,
            },
            message=f"Added a {modifier_type} modifier to '{obj.name}'",
        )
