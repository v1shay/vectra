from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import ensure_object_mode, normalize_optional_string, validate_vector3
from .registry import register_tool

SUPPORTED_PRIMITIVES = {
    "cube": "primitive_cube_add",
    "cylinder": "primitive_cylinder_add",
    "cone": "primitive_cone_add",
    "ico_sphere": "primitive_ico_sphere_add",
    "plane": "primitive_plane_add",
    "torus": "primitive_torus_add",
    "uv_sphere": "primitive_uv_sphere_add",
}

_validate_vector3 = validate_vector3

def _resolve_created_object(context: Any) -> Any:
    for candidate in (
        getattr(context, "active_object", None),
        getattr(getattr(bpy, "context", None), "active_object", None),
        getattr(getattr(bpy, "context", None), "object", None),
    ):
        if candidate is not None:
            return candidate
    return None


@register_tool
class CreatePrimitiveTool(BaseTool):
    name = "mesh.create_primitive"
    description = (
        "Create a new basic shape in the scene, such as a cube or box, cylinder, cone, torus, "
        "flat plane, UV sphere, or ico sphere."
    )
    input_schema = {
        "type": {
            "type": "string",
            "enum": sorted(SUPPORTED_PRIMITIVES),
            "required": False,
        },
        "primitive_type": {
            "type": "string",
            "enum": sorted(SUPPORTED_PRIMITIVES),
            "required": False,
        },
        "name": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
        "scale": {"type": "vector3", "required": False},
        "rotation": {"type": "vector3", "required": False},
        "rotation_euler": {"type": "vector3", "required": False},
    }
    output_schema = {
        "object_name": {"type": "string"},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)

        primitive_type = params.get("type", params.get("primitive_type"))
        if primitive_type not in SUPPORTED_PRIMITIVES:
            raise ToolValidationError(
                f"'type' must be one of {sorted(SUPPORTED_PRIMITIVES)}"
            )

        name = None
        if "name" in params:
            name = normalize_optional_string(params["name"], "name")

        location = (0.0, 0.0, 0.0)
        if "location" in params and params["location"] is not None:
            location = validate_vector3(params["location"], "location")
        scale = None
        if "scale" in params and params["scale"] is not None:
            scale = validate_vector3(params["scale"], "scale")
        rotation = None
        rotation_value = params.get("rotation", params.get("rotation_euler"))
        if rotation_value is not None:
            rotation = validate_vector3(rotation_value, "rotation")

        return {
            "type": primitive_type,
            "name": name,
            "location": location,
            "scale": scale,
            "rotation": rotation,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        validated = self.validate_params(params)
        ensure_object_mode(context)
        operator_name = SUPPORTED_PRIMITIVES[validated["type"]]
        operator = getattr(bpy.ops.mesh, operator_name)
        try:
            result = operator(location=validated["location"])
        except RuntimeError as exc:
            raise ToolExecutionError(
                f"Mesh operator '{operator_name}' failed: {exc}"
            ) from exc
        if "FINISHED" not in result:
            raise ToolExecutionError(f"Mesh operator '{operator_name}' did not finish successfully")

        created_object = _resolve_created_object(context)
        if created_object is None:
            raise ToolExecutionError("Primitive creation did not produce an active object")

        if validated["name"]:
            created_object.name = validated["name"]
        if validated["scale"] is not None:
            created_object.scale = validated["scale"]
        if validated["rotation"] is not None:
            created_object.rotation_euler = validated["rotation"]

        return ToolExecutionResult(
            outputs={"object_name": created_object.name, "object_names": [created_object.name]},
            message=f"Created object '{created_object.name}'",
        )
