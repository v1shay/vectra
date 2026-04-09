from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


_FACE_AXES = {
    "left": (0, -1.0),
    "right": (0, 1.0),
    "back": (1, -1.0),
    "front": (1, 1.0),
    "bottom": (2, -1.0),
    "top": (2, 1.0),
}
_RELATION_TO_FACE = {
    "left_of": "left",
    "right_of": "right",
    "behind": "back",
    "in_front_of": "front",
    "above": "top",
    "below": "bottom",
}
_DEFAULT_PRIMITIVE_EXTENTS = {
    "cube": (2.0, 2.0, 2.0),
    "cylinder": (2.0, 2.0, 2.0),
    "cone": (2.0, 2.0, 2.0),
    "ico_sphere": (2.0, 2.0, 2.0),
    "plane": (2.0, 2.0, 0.0),
    "torus": (2.0, 2.0, 0.5),
    "uv_sphere": (2.0, 2.0, 2.0),
}


def _coerce_vector3(value: Any, *, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    if default is None:
        raise ValueError("Expected a 3-item vector")
    return default


def _mapping_bounds(obj: Mapping[str, Any]) -> dict[str, tuple[float, float, float]] | None:
    raw_bounds = obj.get("bounds")
    if not isinstance(raw_bounds, Mapping):
        return None
    minimum = raw_bounds.get("min")
    maximum = raw_bounds.get("max")
    if not (isinstance(minimum, list) and isinstance(maximum, list) and len(minimum) == 3 and len(maximum) == 3):
        return None
    return {
        "min": _coerce_vector3(minimum),
        "max": _coerce_vector3(maximum),
    }


def _matrix_world_points(obj: Any) -> list[tuple[float, float, float]]:
    bound_box = getattr(obj, "bound_box", None)
    matrix_world = getattr(obj, "matrix_world", None)
    if bound_box is None or matrix_world is None:
        return []

    points: list[tuple[float, float, float]] = []
    for corner in bound_box:
        local_point = _coerce_vector3(corner)
        transformed = None
        try:
            transformed = matrix_world @ local_point
        except Exception:
            try:
                from mathutils import Vector  # type: ignore

                transformed = matrix_world @ Vector(local_point)
            except Exception:
                transformed = None
        if transformed is None:
            return []
        points.append(_coerce_vector3(transformed))
    return points


def _name_of(obj: Any) -> str:
    if isinstance(obj, Mapping):
        name = obj.get("name", "")
    else:
        name = getattr(obj, "name", "")
    return str(name).strip()


def object_type(obj: Any) -> str:
    if isinstance(obj, Mapping):
        raw_type = obj.get("type", "MESH")
    else:
        raw_type = getattr(obj, "type", "MESH")
    normalized = str(raw_type).strip().upper()
    return normalized or "MESH"


def world_bounds(obj: Any) -> dict[str, tuple[float, float, float]]:
    if isinstance(obj, Mapping):
        mapped = _mapping_bounds(obj)
        if mapped is not None:
            return mapped
        location = _coerce_vector3(obj.get("location"), default=(0.0, 0.0, 0.0))
        dimensions = _coerce_vector3(obj.get("dimensions"), default=(2.0, 2.0, 2.0))
    else:
        points = _matrix_world_points(obj)
        if points:
            minimum = (
                min(point[0] for point in points),
                min(point[1] for point in points),
                min(point[2] for point in points),
            )
            maximum = (
                max(point[0] for point in points),
                max(point[1] for point in points),
                max(point[2] for point in points),
            )
            return {"min": minimum, "max": maximum}
        location = _coerce_vector3(getattr(obj, "location", None), default=(0.0, 0.0, 0.0))
        dimensions = _coerce_vector3(getattr(obj, "dimensions", None), default=(2.0, 2.0, 2.0))

    half_extents = tuple(float(component) / 2.0 for component in dimensions)
    return {
        "min": tuple(location[index] - half_extents[index] for index in range(3)),
        "max": tuple(location[index] + half_extents[index] for index in range(3)),
    }


def primitive_bounds(
    primitive_type: str,
    *,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] | None = None,
) -> dict[str, tuple[float, float, float]]:
    extents = _DEFAULT_PRIMITIVE_EXTENTS.get(str(primitive_type).strip().lower(), (2.0, 2.0, 2.0))
    if scale is not None:
        extents = tuple(float(extents[index]) * abs(float(scale[index])) for index in range(3))
    half_extents = tuple(float(component) / 2.0 for component in extents)
    return {
        "min": tuple(location[index] - half_extents[index] for index in range(3)),
        "max": tuple(location[index] + half_extents[index] for index in range(3)),
    }


def bounds_center(bounds: Mapping[str, Any]) -> tuple[float, float, float]:
    minimum = _coerce_vector3(bounds.get("min"))
    maximum = _coerce_vector3(bounds.get("max"))
    return tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))


def bounds_extents(bounds: Mapping[str, Any]) -> tuple[float, float, float]:
    minimum = _coerce_vector3(bounds.get("min"))
    maximum = _coerce_vector3(bounds.get("max"))
    return tuple(maximum[index] - minimum[index] for index in range(3))


def bounds_half_extents(bounds: Mapping[str, Any]) -> tuple[float, float, float]:
    extents = bounds_extents(bounds)
    return tuple(float(component) / 2.0 for component in extents)


def face_center(bounds: Mapping[str, Any], face: str) -> tuple[float, float, float]:
    normalized_face = str(face).strip().lower()
    if normalized_face not in _FACE_AXES:
        raise ValueError(f"Unsupported face '{face}'")

    center = list(bounds_center(bounds))
    minimum = _coerce_vector3(bounds.get("min"))
    maximum = _coerce_vector3(bounds.get("max"))
    axis_index, direction = _FACE_AXES[normalized_face]
    center[axis_index] = maximum[axis_index] if direction > 0.0 else minimum[axis_index]
    return (center[0], center[1], center[2])


def lowest_z(obj: Any) -> float:
    return float(world_bounds(obj)["min"][2])


def place_on_surface_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    surface: str = "top",
    offset: float = 0.0,
) -> tuple[float, float, float]:
    normalized_surface = str(surface).strip().lower()
    if normalized_surface not in _FACE_AXES:
        raise ValueError(f"Unsupported surface '{surface}'")

    reference_center = list(face_center(reference_bounds, normalized_surface))
    target_half_extents = bounds_half_extents(target_bounds)
    axis_index, direction = _FACE_AXES[normalized_surface]
    reference_center[axis_index] += direction * (target_half_extents[axis_index] + float(offset))
    return (reference_center[0], reference_center[1], reference_center[2])


def place_against_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    side: str,
    offset: float = 0.0,
) -> tuple[float, float, float]:
    return place_on_surface_location(
        target_bounds,
        reference_bounds,
        surface=side,
        offset=offset,
    )


def place_relative_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    relation: str,
    distance: float,
) -> tuple[float, float, float]:
    normalized_relation = str(relation).strip().lower()
    if normalized_relation not in _RELATION_TO_FACE:
        raise ValueError(f"Unsupported relation '{relation}'")
    return place_on_surface_location(
        target_bounds,
        reference_bounds,
        surface=_RELATION_TO_FACE[normalized_relation],
        offset=float(distance),
    )


def align_to_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    axis: str,
) -> tuple[float, float, float]:
    normalized_axis = str(axis).strip().lower()
    if normalized_axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported axis '{axis}'")

    axis_index = {"x": 0, "y": 1, "z": 2}[normalized_axis]
    target_center = list(bounds_center(target_bounds))
    reference_center = bounds_center(reference_bounds)
    target_center[axis_index] = reference_center[axis_index]
    return (target_center[0], target_center[1], target_center[2])


def center_for_lowest_z(bounds: Mapping[str, Any], target_lowest_z: float) -> tuple[float, float, float]:
    center = list(bounds_center(bounds))
    half_extents = bounds_half_extents(bounds)
    center[2] = float(target_lowest_z) + half_extents[2]
    return (center[0], center[1], center[2])


def is_floor_like(obj: Any) -> bool:
    if object_type(obj) != "MESH":
        return False

    name = _name_of(obj).lower()
    if "floor" in name:
        return True

    extents = bounds_extents(world_bounds(obj))
    return extents[2] <= 0.25 and extents[0] >= 2.0 and extents[1] >= 2.0


def find_floor_object(objects: Iterable[Any]) -> Any | None:
    named_candidates: list[Any] = []
    flat_candidates: list[Any] = []
    for obj in objects:
        if not is_floor_like(obj):
            continue
        if "floor" in _name_of(obj).lower():
            named_candidates.append(obj)
        else:
            flat_candidates.append(obj)

    if named_candidates:
        return sorted(named_candidates, key=lambda candidate: _name_of(candidate).lower())[0]
    if flat_candidates:
        return sorted(flat_candidates, key=lambda candidate: (_name_of(candidate).lower(), lowest_z(candidate)))[0]
    return None
