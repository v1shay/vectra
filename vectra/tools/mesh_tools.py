from __future__ import annotations

from collections.abc import Mapping
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

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _validate_vector3(
    value: Any,
    field_name: str,
    *,
    default: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    if value is None:
        if default is not None:
            return default
        raise ToolValidationError(f"'{field_name}' must be a 3-item list or tuple")

    if isinstance(value, Mapping):
        if not value:
            if default is not None:
                return default
            raise ToolValidationError(f"'{field_name}' mapping must include at least one axis")

        unexpected_axes = sorted(set(value) - set(AXIS_INDEX))
        if unexpected_axes:
            raise ToolValidationError(
                f"'{field_name}' mapping may only use x, y, and z axes"
            )

        base = list(default) if default is not None else [None, None, None]
        for axis, raw_component in value.items():
            try:
                base[AXIS_INDEX[axis]] = float(raw_component)
            except (TypeError, ValueError) as exc:
                raise ToolValidationError(f"'{field_name}' values must be numeric") from exc

        if any(component is None for component in base):
            raise ToolValidationError(
                f"'{field_name}' must include all x, y, and z values when no default is available"
            )

        return (float(base[0]), float(base[1]), float(base[2]))

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ToolValidationError(
            f"'{field_name}' must be a 3-item list or tuple, or an axis mapping"
        )
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"'{field_name}' values must be numeric") from exc


def _normalize_optional_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError("'name' must be a string when provided")

    normalized = value.strip()
    return normalized or None


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
        "primitive_type": {"type": "string", "enum": sorted(SUPPORTED_PRIMITIVES)},
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

        name = _normalize_optional_name(params.get("name"))
        location = _validate_vector3(
            params.get("location"),
            "location",
            default=(0.0, 0.0, 0.0),
        )

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
