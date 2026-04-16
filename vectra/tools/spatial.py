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
    "next_to": "right",
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
_CONTACT_TOLERANCE = 0.05
_NEAR_TOLERANCE = 1.0
_FACE_NAMES = ("left", "right", "back", "front", "bottom", "top")


def _coerce_vector3(value: Any, *, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    if not isinstance(value, (str, bytes, bytearray)):
        try:
            components = list(value)
        except TypeError:
            components = None
        if components is not None and len(components) == 3:
            return (float(components[0]), float(components[1]), float(components[2]))
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


def all_face_centers(bounds: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    return {face: face_center(bounds, face) for face in _FACE_NAMES}


def bounds_to_lists(bounds: Mapping[str, Any]) -> dict[str, list[float]]:
    minimum = _coerce_vector3(bounds.get("min"))
    maximum = _coerce_vector3(bounds.get("max"))
    return {
        "min": [float(component) for component in minimum],
        "max": [float(component) for component in maximum],
    }


def vector_to_list(vector: Any) -> list[float]:
    coerced = _coerce_vector3(vector)
    return [float(component) for component in coerced]


def face_centers_to_lists(bounds: Mapping[str, Any]) -> dict[str, list[float]]:
    return {face: vector_to_list(center) for face, center in all_face_centers(bounds).items()}


def interval_gap(
    first_min: float,
    first_max: float,
    second_min: float,
    second_max: float,
) -> float:
    if first_max < second_min:
        return float(second_min - first_max)
    if second_max < first_min:
        return float(first_min - second_max)
    overlap = min(first_max, second_max) - max(first_min, second_min)
    return -float(overlap)


def bounds_axis_gap(
    first_bounds: Mapping[str, Any],
    second_bounds: Mapping[str, Any],
    *,
    axis: int,
) -> float:
    first_minimum = _coerce_vector3(first_bounds.get("min"))
    first_maximum = _coerce_vector3(first_bounds.get("max"))
    second_minimum = _coerce_vector3(second_bounds.get("min"))
    second_maximum = _coerce_vector3(second_bounds.get("max"))
    return interval_gap(
        first_minimum[axis],
        first_maximum[axis],
        second_minimum[axis],
        second_maximum[axis],
    )


def bounds_overlap_on_axes(
    first_bounds: Mapping[str, Any],
    second_bounds: Mapping[str, Any],
    axes: Iterable[int],
    *,
    tolerance: float = _CONTACT_TOLERANCE,
) -> bool:
    return all(
        bounds_axis_gap(first_bounds, second_bounds, axis=axis) <= float(tolerance)
        for axis in axes
    )


def face_gap(
    source_bounds: Mapping[str, Any],
    target_bounds: Mapping[str, Any],
    *,
    source_face: str,
    target_face: str,
) -> float:
    source_face_name = str(source_face).strip().lower()
    target_face_name = str(target_face).strip().lower()
    if source_face_name not in _FACE_AXES:
        raise ValueError(f"Unsupported source face '{source_face}'")
    if target_face_name not in _FACE_AXES:
        raise ValueError(f"Unsupported target face '{target_face}'")
    source_axis, _ = _FACE_AXES[source_face_name]
    target_axis, _ = _FACE_AXES[target_face_name]
    if source_axis != target_axis:
        raise ValueError("Face gap requires faces on the same axis")
    source_center = face_center(source_bounds, source_face_name)
    target_center = face_center(target_bounds, target_face_name)
    return abs(float(source_center[source_axis]) - float(target_center[target_axis]))


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


def is_wall_like(obj: Any) -> bool:
    if object_type(obj) != "MESH":
        return False

    extents = bounds_extents(world_bounds(obj))
    horizontal_extents = (float(extents[0]), float(extents[1]))
    return (
        float(extents[2]) >= 1.0
        and min(horizontal_extents) <= 0.35
        and max(horizontal_extents) >= 1.0
    )


def is_grounded_on(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    tolerance: float = _CONTACT_TOLERANCE,
) -> bool:
    target_minimum = _coerce_vector3(target_bounds.get("min"))
    reference_maximum = _coerce_vector3(reference_bounds.get("max"))
    return (
        abs(float(target_minimum[2]) - float(reference_maximum[2])) <= float(tolerance)
        and bounds_overlap_on_axes(target_bounds, reference_bounds, (0, 1), tolerance=tolerance)
    )


def floor_contact_record(
    target: Mapping[str, Any],
    floor_candidates: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = _CONTACT_TOLERANCE,
) -> dict[str, Any] | None:
    target_name = str(target.get("name", "")).strip()
    target_bounds = world_bounds(target)
    contacts: list[dict[str, Any]] = []
    for candidate in floor_candidates:
        floor_name = str(candidate.get("name", "")).strip()
        if not floor_name or floor_name == target_name:
            continue
        floor_bounds = world_bounds(candidate)
        if not is_grounded_on(target_bounds, floor_bounds, tolerance=tolerance):
            continue
        gap = face_gap(target_bounds, floor_bounds, source_face="bottom", target_face="top")
        contacts.append(
            {
                "object": floor_name,
                "gap": float(gap),
            }
        )
    if not contacts:
        return None
    return sorted(contacts, key=lambda item: (item["gap"], item["object"]))[0]


def classify_spatial_relation(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    contact_tolerance: float = _CONTACT_TOLERANCE,
    near_tolerance: float = _NEAR_TOLERANCE,
) -> list[str]:
    source_bounds = world_bounds(source)
    target_bounds = world_bounds(target)
    source_center = bounds_center(source_bounds)
    target_center = bounds_center(target_bounds)
    relations: list[str] = []

    xy_overlap = bounds_overlap_on_axes(source_bounds, target_bounds, (0, 1), tolerance=contact_tolerance)
    xz_overlap = bounds_overlap_on_axes(source_bounds, target_bounds, (0, 2), tolerance=contact_tolerance)
    yz_overlap = bounds_overlap_on_axes(source_bounds, target_bounds, (1, 2), tolerance=contact_tolerance)

    if is_grounded_on(source_bounds, target_bounds, tolerance=contact_tolerance):
        relations.append("on")

    if xy_overlap:
        z_gap = bounds_axis_gap(source_bounds, target_bounds, axis=2)
        if source_center[2] > target_center[2] and z_gap <= near_tolerance:
            relations.append("above")
        elif source_center[2] < target_center[2] and z_gap <= near_tolerance:
            relations.append("below")

    x_gap = bounds_axis_gap(source_bounds, target_bounds, axis=0)
    if yz_overlap and x_gap <= near_tolerance:
        if abs(x_gap) <= contact_tolerance:
            relations.append("against")
        relations.append("left_of" if source_center[0] < target_center[0] else "right_of")
        relations.append("next_to")

    y_gap = bounds_axis_gap(source_bounds, target_bounds, axis=1)
    if xz_overlap and y_gap <= near_tolerance:
        if abs(y_gap) <= contact_tolerance:
            relations.append("against")
        relations.append("behind" if source_center[1] < target_center[1] else "in_front_of")
        relations.append("next_to")

    deduped: list[str] = []
    for relation in relations:
        if relation not in deduped:
            deduped.append(relation)
    return deduped


def spatial_relations(
    objects: Iterable[Mapping[str, Any]],
    *,
    contact_tolerance: float = _CONTACT_TOLERANCE,
    near_tolerance: float = _NEAR_TOLERANCE,
) -> list[dict[str, Any]]:
    records = sorted(
        [obj for obj in objects if isinstance(obj.get("name"), str) and obj.get("bounds")],
        key=lambda item: str(item.get("name", "")),
    )
    relations: list[dict[str, Any]] = []
    for source in records:
        source_name = str(source.get("name", "")).strip()
        for target in records:
            target_name = str(target.get("name", "")).strip()
            if not source_name or not target_name or source_name == target_name:
                continue
            for relation in classify_spatial_relation(
                source,
                target,
                contact_tolerance=contact_tolerance,
                near_tolerance=near_tolerance,
            ):
                relations.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "relation": relation,
                    }
                )
    return sorted(
        relations,
        key=lambda item: (item["source"], item["target"], item["relation"]),
    )


def spatial_metadata_for_object(
    obj: Mapping[str, Any],
    *,
    floor_candidates: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    bounds = world_bounds(obj)
    center = bounds_center(bounds)
    extents = bounds_extents(bounds)
    half_extents = bounds_half_extents(bounds)
    floor_contact = floor_contact_record(obj, floor_candidates)
    return {
        "center": vector_to_list(center),
        "extents": vector_to_list(extents),
        "half_extents": vector_to_list(half_extents),
        "face_centers": face_centers_to_lists(bounds),
        "grounded": floor_contact is not None or abs(float(bounds["min"][2])) <= _CONTACT_TOLERANCE,
        "floor_contact": floor_contact,
        "is_floor_like": is_floor_like(obj),
        "is_wall_like": is_wall_like(obj),
    }


def _scene_bounds_from_records(records: list[Mapping[str, Any]]) -> dict[str, tuple[float, float, float]] | None:
    if not records:
        return None
    bounds = [world_bounds(record) for record in records]
    return {
        "min": tuple(min(float(item["min"][index]) for item in bounds) for index in range(3)),
        "max": tuple(max(float(item["max"][index]) for item in bounds) for index in range(3)),
    }


def _wall_inner_face(wall_bounds: Mapping[str, Any], scene_bounds: Mapping[str, Any]) -> str | None:
    extents = bounds_extents(wall_bounds)
    horizontal = [(0, extents[0]), (1, extents[1])]
    thin_axis, thin_extent = min(horizontal, key=lambda item: item[1])
    if float(thin_extent) > 0.35:
        return None
    wall_center = bounds_center(wall_bounds)
    scene_center = bounds_center(scene_bounds)
    if thin_axis == 0:
        return "right" if wall_center[0] <= scene_center[0] else "left"
    return "front" if wall_center[1] <= scene_center[1] else "back"


def spatial_anchors(objects: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = sorted(
        [obj for obj in objects if isinstance(obj.get("name"), str) and obj.get("bounds")],
        key=lambda item: str(item.get("name", "")),
    )
    scene_bounds = _scene_bounds_from_records(records)
    anchors: list[dict[str, Any]] = []

    for obj in records:
        name = str(obj.get("name", "")).strip()
        bounds = world_bounds(obj)
        for face, center in all_face_centers(bounds).items():
            anchors.append(
                {
                    "name": f"{name}.face.{face}",
                    "type": "object_face",
                    "object": name,
                    "face": face,
                    "location": vector_to_list(center),
                }
            )

        if is_floor_like(obj):
            anchors.append(
                {
                    "name": f"{name}.floor.top",
                    "type": "floor_top",
                    "object": name,
                    "location": vector_to_list(face_center(bounds, "top")),
                }
            )
            anchors.append(
                {
                    "name": f"{name}.floor.center",
                    "type": "floor_center",
                    "object": name,
                    "location": vector_to_list(bounds_center(bounds)),
                }
            )

        if scene_bounds is not None and is_wall_like(obj):
            inner_face = _wall_inner_face(bounds, scene_bounds)
            if inner_face:
                anchors.append(
                    {
                        "name": f"{name}.wall.inner",
                        "type": "wall_inner_face",
                        "object": name,
                        "face": inner_face,
                        "location": vector_to_list(face_center(bounds, inner_face)),
                    }
                )

    if scene_bounds is not None:
        minimum = _coerce_vector3(scene_bounds.get("min"))
        maximum = _coerce_vector3(scene_bounds.get("max"))
        for x_name, x_value in (("min_x", minimum[0]), ("max_x", maximum[0])):
            for y_name, y_value in (("min_y", minimum[1]), ("max_y", maximum[1])):
                anchors.append(
                    {
                        "name": f"scene.corner.{x_name}.{y_name}",
                        "type": "scene_corner",
                        "location": [float(x_value), float(y_value), float(minimum[2])],
                    }
                )

    return sorted(anchors, key=lambda item: item["name"])


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
