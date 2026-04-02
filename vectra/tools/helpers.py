from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import ToolExecutionError, ToolValidationError


def validate_vector3(value: Any, field_name: str) -> tuple[float, float, float]:
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


def normalize_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError(f"'{field_name}' must be a string when provided")
    normalized = value.strip()
    if not normalized:
        raise ToolValidationError(f"'{field_name}' must be a non-empty string when provided")
    return normalized


def current_mode(context: Any) -> str | None:
    if bpy is None:
        return None
    for candidate in (context, getattr(bpy, "context", None)):
        mode = getattr(candidate, "mode", None)
        if isinstance(mode, str):
            return mode
    return None


def ensure_object_mode(context: Any) -> None:
    if bpy is None:
        raise ToolExecutionError("Blender Python API is unavailable")
    mode = current_mode(context)
    if mode in {None, "OBJECT"}:
        return
    mode_set = getattr(getattr(bpy.ops, "object", None), "mode_set", None)
    if mode_set is None:
        raise ToolExecutionError(f"Cannot switch Blender mode from '{mode}'")
    result = mode_set(mode="OBJECT")
    if isinstance(result, set) and "FINISHED" not in result:
        raise ToolExecutionError(f"Failed to switch Blender mode: {result}")


def active_scene(context: Any) -> Any:
    if bpy is None:
        raise ToolExecutionError("Blender Python API is unavailable")
    scene = getattr(context, "scene", None) or getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        raise ToolExecutionError("No active Blender scene is available")
    return scene


def scene_objects(context: Any) -> list[Any]:
    scene = active_scene(context)
    return list(getattr(scene, "objects", []))


def active_object(context: Any) -> Any:
    if bpy is None:
        return None
    return (
        getattr(context, "active_object", None)
        or getattr(getattr(bpy, "context", None), "active_object", None)
        or getattr(getattr(bpy, "context", None), "object", None)
    )


def selected_objects(context: Any) -> list[Any]:
    if bpy is None:
        return []
    selected = getattr(context, "selected_objects", None)
    if isinstance(selected, list):
        return list(selected)
    selected = getattr(getattr(bpy, "context", None), "selected_objects", None)
    if isinstance(selected, list):
        return list(selected)
    return []


def resolve_object(context: Any, target: Any) -> Any:
    if bpy is None:
        raise ToolExecutionError("Blender Python API is unavailable")
    objects = scene_objects(context)
    if isinstance(target, str) and target.strip():
        stripped = target.strip()
        for obj in objects:
            if getattr(obj, "name", None) == stripped:
                return obj
        lowered = stripped.lower()
        for obj in objects:
            name = getattr(obj, "name", "")
            if isinstance(name, str) and (name.lower() == lowered or lowered in name.lower()):
                return obj

    candidate = active_object(context)
    if candidate is not None:
        return candidate

    selected = selected_objects(context)
    if selected:
        return selected[0]

    if objects:
        return objects[-1]
    return None


def resolve_objects(context: Any, targets: Any) -> list[Any]:
    if isinstance(targets, list):
        resolved = [resolve_object(context, target) for target in targets]
        return [obj for obj in resolved if obj is not None]
    target = resolve_object(context, targets)
    return [target] if target is not None else []


def vector_to_list(vector: Any) -> list[float]:
    return [float(component) for component in vector[:3]]


def default_spacing_for_object(obj: Any) -> float:
    dimensions = getattr(obj, "dimensions", (2.0, 2.0, 2.0))
    return max(float(dimensions[0]), 1.0) + 0.5


def look_at_rotation(source: list[float], target: list[float]) -> tuple[float, float, float]:
    dx = float(target[0]) - float(source[0])
    dy = float(target[1]) - float(source[1])
    dz = float(target[2]) - float(source[2])
    horizontal = max((dx * dx + dy * dy) ** 0.5, 0.0001)
    yaw = __import__("math").atan2(dx, dy)
    pitch = __import__("math").atan2(-dz, horizontal)
    return (pitch, 0.0, yaw)
