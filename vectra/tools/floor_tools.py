from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult
from .helpers import ensure_object_mode, scene_objects
from .registry import register_tool
from .spatial import bounds_extents, center_for_lowest_z, find_floor_object, object_type, world_bounds


def _active_object() -> Any:
    if bpy is None:
        return None
    return getattr(getattr(bpy, "context", None), "active_object", None)


def _resolved_bounds_payload(obj: Any) -> dict[str, list[float]]:
    bounds = world_bounds(obj)
    return {
        "min": [float(component) for component in bounds["min"]],
        "max": [float(component) for component in bounds["max"]],
    }


def _mesh_objects(context: Any) -> list[Any]:
    return [obj for obj in scene_objects(context) if object_type(obj) == "MESH"]


def _scene_floor_scale(mesh_objects: list[Any]) -> tuple[float, float, float]:
    if not mesh_objects:
        return (4.0, 4.0, 1.0)

    minimum = [float("inf"), float("inf")]
    maximum = [float("-inf"), float("-inf")]
    for obj in mesh_objects:
        bounds = world_bounds(obj)
        minimum[0] = min(minimum[0], float(bounds["min"][0]))
        minimum[1] = min(minimum[1], float(bounds["min"][1]))
        maximum[0] = max(maximum[0], float(bounds["max"][0]))
        maximum[1] = max(maximum[1], float(bounds["max"][1]))

    span_x = max(maximum[0] - minimum[0], 0.0)
    span_y = max(maximum[1] - minimum[1], 0.0)
    floor_size = max(span_x, span_y, 8.0) + 2.0
    scale_value = floor_size / 2.0
    return (scale_value, scale_value, 1.0)


def _normalized_floor_location(obj: Any) -> tuple[float, float, float]:
    bounds = world_bounds(obj)
    return center_for_lowest_z(bounds, 0.0)


@register_tool
class EnsureFloorTool(BaseTool):
    name = "scene.ensure_floor"
    description = "Ensure the scene has a floor-like object normalized to rest on z=0."
    input_schema = {}

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        del params
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")

        ensure_object_mode(context)
        mesh_objects = _mesh_objects(context)
        floor_object = find_floor_object(mesh_objects)
        if floor_object is not None:
            floor_object.location = _normalized_floor_location(floor_object)
            return ToolExecutionResult(
                outputs={
                    "object_name": floor_object.name,
                    "object_names": [floor_object.name],
                    "normalized_existing": True,
                    "created_new": False,
                    "resolved_location": [float(component) for component in floor_object.location[:3]],
                    "resolved_bounds": _resolved_bounds_payload(floor_object),
                },
                message=f"Normalized floor '{floor_object.name}' to z=0",
            )

        result = bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
        if isinstance(result, set) and "FINISHED" not in result:
            raise ToolExecutionError(f"Failed to create a floor plane: {result}")

        floor_object = _active_object()
        if floor_object is None:
            raise ToolExecutionError("Floor creation did not produce an active object")

        floor_object.name = "Floor"
        floor_object.scale = _scene_floor_scale(mesh_objects)
        floor_object.location = _normalized_floor_location(floor_object)

        return ToolExecutionResult(
            outputs={
                "object_name": floor_object.name,
                "object_names": [floor_object.name],
                "normalized_existing": False,
                "created_new": True,
                "resolved_location": [float(component) for component in floor_object.location[:3]],
                "resolved_bounds": _resolved_bounds_payload(floor_object),
            },
            message=f"Created floor '{floor_object.name}' at z=0",
        )
