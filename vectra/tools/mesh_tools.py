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

    normalized: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise ToolValidationError(f"'{field_name}' values must be numeric")
        try:
            normalized.append(float(component))
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"'{field_name}' values must be numeric") from exc
    return (normalized[0], normalized[1], normalized[2])


def _normalize_optional_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError("'name' must be a string when provided")

    normalized = value.strip()
    if not normalized:
        raise ToolValidationError("'name' must be a non-empty string when provided")
    return normalized


def _current_mode(context: Any) -> str | None:
    for candidate in (context, getattr(bpy, "context", None)):
        mode = getattr(candidate, "mode", None)
        if isinstance(mode, str):
            return mode
    return None


def _ensure_object_mode(context: Any) -> None:
    current_mode = _current_mode(context)
    if current_mode is None or current_mode == "OBJECT":
        return

    object_ops = getattr(getattr(bpy, "ops", None), "object", None)
    mode_set = getattr(object_ops, "mode_set", None)
    if mode_set is None:
        raise ToolExecutionError(
            f"Cannot switch Blender mode from '{current_mode}' before creating a primitive"
        )

    try:
        result = mode_set(mode="OBJECT")
    except RuntimeError as exc:
        raise ToolExecutionError(
            f"Failed to switch Blender to Object Mode before primitive creation: {exc}"
        ) from exc

    if isinstance(result, set) and "FINISHED" not in result:
        raise ToolExecutionError(
            f"Failed to switch Blender to Object Mode before primitive creation: {result}"
        )


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
        "Create a new basic shape in the scene, such as a cube or box, a flat plane or square "
        "surface, or a UV sphere for spheres, balls, and other round 3D shapes."
    )
    input_schema = {
        "primitive_type": {
            "type": "string",
            "enum": sorted(SUPPORTED_PRIMITIVES),
            "required": True,
        },
        "name": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
    }
    output_schema = {
        "object_name": {"type": "string"},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)

        primitive_type = params.get("primitive_type")
        if primitive_type not in SUPPORTED_PRIMITIVES:
            raise ToolValidationError(
                f"'primitive_type' must be one of {sorted(SUPPORTED_PRIMITIVES)}"
            )

        name = None
        if "name" in params:
            name = _normalize_optional_name(params["name"])

        location = (0.0, 0.0, 0.0)
        if "location" in params:
            location = _validate_vector3(params["location"], "location")

        return {
            "primitive_type": primitive_type,
            "name": name,
            "location": location,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        validated = self.validate_params(params)
        _ensure_object_mode(context)
        operator_name = SUPPORTED_PRIMITIVES[validated["primitive_type"]]
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

        return ToolExecutionResult(
            outputs={"object_name": created_object.name},
            message=f"Created object '{created_object.name}'",
        )
