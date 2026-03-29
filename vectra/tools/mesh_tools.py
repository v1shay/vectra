from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .registry import register_tool

SUPPORTED_PRIMITIVES = {
    "cube": "primitive_cube_add",
    "plane": "primitive_plane_add",
    "uv_sphere": "primitive_uv_sphere_add",
}


def _validate_vector3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ToolValidationError(f"'{field_name}' must be a 3-item list or tuple")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"'{field_name}' values must be numeric") from exc


@register_tool
class CreatePrimitiveTool(BaseTool):
    name = "mesh.create_primitive"
    description = "Create a supported mesh primitive"
    input_schema = {
        "primitive_type": {"type": "string", "enum": sorted(SUPPORTED_PRIMITIVES)},
        "name": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)

        primitive_type = params.get("primitive_type")
        if primitive_type not in SUPPORTED_PRIMITIVES:
            raise ToolValidationError(
                f"'primitive_type' must be one of {sorted(SUPPORTED_PRIMITIVES)}"
            )

        name = params.get("name")
        if name is not None and not isinstance(name, str):
            raise ToolValidationError("'name' must be a string when provided")

        location = _validate_vector3(params.get("location", [0.0, 0.0, 0.0]), "location")

        return {
            "primitive_type": primitive_type,
            "name": name,
            "location": location,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        validated = self.validate_params(params)
        operator_name = SUPPORTED_PRIMITIVES[validated["primitive_type"]]
        operator = getattr(bpy.ops.mesh, operator_name)
        result = operator(location=validated["location"])
        if "FINISHED" not in result:
            raise ToolExecutionError(f"Mesh operator '{operator_name}' did not finish successfully")

        created_object = getattr(context, "active_object", None)
        if created_object is None:
            raise ToolExecutionError("Primitive creation did not produce an active object")

        if validated["name"]:
            created_object.name = validated["name"]

        return ToolExecutionResult(
            outputs={"object_name": created_object.name},
            message=f"Created object '{created_object.name}'",
        )
