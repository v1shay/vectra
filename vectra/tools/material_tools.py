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
class ApplyBasicMaterialTool(BaseTool):
    name = "material.apply_basic"
    description = "Apply a simple material tweak to an object."
    input_schema = {
        "target": {"type": "string", "required": False},
        "color": {"type": "vector3", "required": False},
        "roughness": {"type": "number", "required": False},
        "metallic": {"type": "number", "required": False},
    }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        obj = resolve_object(context, params.get("target"))
        if obj is None:
            raise ToolExecutionError("No material target could be resolved")
        if getattr(obj, "type", "") != "MESH":
            raise ToolExecutionError(f"Object '{obj.name}' does not support mesh materials")

        material = obj.active_material
        if material is None:
            material = bpy.data.materials.new(name=f"{obj.name}_Material")
            obj.active_material = material
        material.use_nodes = True
        principled = next(
            (node for node in material.node_tree.nodes if getattr(node, "type", "") == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            raise ToolExecutionError("The material node tree is missing a Principled BSDF node")

        if "color" in params and params["color"] is not None:
            color = validate_vector3(params["color"], "color")
            principled.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        if "roughness" in params and params["roughness"] is not None:
            if isinstance(params["roughness"], bool):
                raise ToolValidationError("'roughness' must be numeric")
            principled.inputs["Roughness"].default_value = float(params["roughness"])
        if "metallic" in params and params["metallic"] is not None:
            if isinstance(params["metallic"], bool):
                raise ToolValidationError("'metallic' must be numeric")
            principled.inputs["Metallic"].default_value = float(params["metallic"])

        return ToolExecutionResult(
            outputs={"object_name": obj.name, "object_names": [obj.name]},
            message=f"Applied a basic material to '{obj.name}'",
        )
