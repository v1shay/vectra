from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


Vector3 = tuple[float, float, float]
Bounds = dict[str, Vector3]

FACE_AXES: dict[str, tuple[int, float]] = {
    "left": (0, -1.0),
    "right": (0, 1.0),
    "back": (1, -1.0),
    "front": (1, 1.0),
    "bottom": (2, -1.0),
    "top": (2, 1.0),
}
FACE_NAMES = ("left", "right", "back", "front", "bottom", "top")
RELATION_TO_FACE = {
    "left_of": "left",
    "right_of": "right",
    "next_to": "right",
    "behind": "back",
    "in_front_of": "front",
    "above": "top",
    "below": "bottom",
}
AXIS_NAMES = {"x": 0, "y": 1, "z": 2}

CONTACT_TOLERANCE = 0.05
NEAR_TOLERANCE = 1.0

_DEFAULT_PRIMITIVE_EXTENTS: dict[str, Vector3] = {
    "cube": (2.0, 2.0, 2.0),
    "cylinder": (2.0, 2.0, 2.0),
    "cone": (2.0, 2.0, 2.0),
    "ico_sphere": (2.0, 2.0, 2.0),
    "plane": (2.0, 2.0, 0.0),
    "torus": (2.0, 2.0, 0.5),
    "uv_sphere": (2.0, 2.0, 2.0),
}


def _finite_float(value: Any, *, field_name: str = "value") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def coerce_vector3(
    value: Any,
    *,
    default: Vector3 | None = None,
    field_name: str = "vector",
) -> Vector3:
    if not isinstance(value, (str, bytes, bytearray)):
        try:
            components = list(value)
        except TypeError:
            components = None
        if components is not None and len(components) == 3:
            return (
                _finite_float(components[0], field_name=field_name),
                _finite_float(components[1], field_name=field_name),
                _finite_float(components[2], field_name=field_name),
            )
    if default is None:
        raise ValueError(f"{field_name} must be a 3-item vector")
    return default


def vector_to_list(vector: Any) -> list[float]:
    return [float(component) for component in coerce_vector3(vector)]


def _mapping_bounds(obj: Mapping[str, Any]) -> Bounds | None:
    for key in ("world_bounds", "bounds"):
        raw_bounds = obj.get(key)
        if not isinstance(raw_bounds, Mapping):
            continue
        minimum = raw_bounds.get("min")
        maximum = raw_bounds.get("max")
        if minimum is None or maximum is None:
            continue
        return {
            "min": coerce_vector3(minimum, field_name=f"{key}.min"),
            "max": coerce_vector3(maximum, field_name=f"{key}.max"),
        }
    return None


def _matrix_world_points(obj: Any) -> list[Vector3]:
    bound_box = getattr(obj, "bound_box", None)
    matrix_world = getattr(obj, "matrix_world", None)
    if bound_box is None or matrix_world is None:
        return []

    points: list[Vector3] = []
    for corner in bound_box:
        local_point = coerce_vector3(corner, field_name="bound_box")
        transformed: Any = None
        try:
            from mathutils import Vector  # type: ignore

            transformed = matrix_world @ Vector(local_point)
        except Exception:
            try:
                transformed = matrix_world @ local_point
            except Exception:
                return []
        points.append(coerce_vector3(transformed, field_name="matrix_world point"))
    return points


def name_of(obj: Any) -> str:
    if isinstance(obj, Mapping):
        raw_name = obj.get("name", "")
    else:
        raw_name = getattr(obj, "name", "")
    return str(raw_name).strip()


def object_type(obj: Any) -> str:
    if isinstance(obj, Mapping):
        raw_type = obj.get("type", "MESH")
    else:
        raw_type = getattr(obj, "type", "MESH")
    normalized = str(raw_type).strip().upper()
    return normalized or "MESH"


def world_bounds(obj: Any) -> Bounds:
    if isinstance(obj, Mapping):
        mapped = _mapping_bounds(obj)
        if mapped is not None:
            return mapped
        location = coerce_vector3(obj.get("location"), default=(0.0, 0.0, 0.0), field_name="location")
        dimensions = coerce_vector3(obj.get("dimensions"), default=(2.0, 2.0, 2.0), field_name="dimensions")
    else:
        points = _matrix_world_points(obj)
        if points:
            return {
                "min": tuple(min(point[index] for point in points) for index in range(3)),
                "max": tuple(max(point[index] for point in points) for index in range(3)),
            }
        location = coerce_vector3(getattr(obj, "location", None), default=(0.0, 0.0, 0.0), field_name="location")
        dimensions = coerce_vector3(getattr(obj, "dimensions", None), default=(2.0, 2.0, 2.0), field_name="dimensions")

    half_extents = tuple(abs(float(component)) / 2.0 for component in dimensions)
    return {
        "min": tuple(location[index] - half_extents[index] for index in range(3)),
        "max": tuple(location[index] + half_extents[index] for index in range(3)),
    }


def primitive_bounds(
    primitive_type: str,
    *,
    location: Vector3 = (0.0, 0.0, 0.0),
    scale: Vector3 | None = None,
) -> Bounds:
    extents = _DEFAULT_PRIMITIVE_EXTENTS.get(str(primitive_type).strip().lower(), (2.0, 2.0, 2.0))
    if scale is not None:
        extents = tuple(abs(extents[index] * float(scale[index])) for index in range(3))
    half_extents = tuple(float(component) / 2.0 for component in extents)
    return {
        "min": tuple(float(location[index]) - half_extents[index] for index in range(3)),
        "max": tuple(float(location[index]) + half_extents[index] for index in range(3)),
    }


def bounds_center(bounds: Mapping[str, Any]) -> Vector3:
    minimum = coerce_vector3(bounds.get("min"), field_name="bounds.min")
    maximum = coerce_vector3(bounds.get("max"), field_name="bounds.max")
    return tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))


def bounds_extents(bounds: Mapping[str, Any]) -> Vector3:
    minimum = coerce_vector3(bounds.get("min"), field_name="bounds.min")
    maximum = coerce_vector3(bounds.get("max"), field_name="bounds.max")
    return tuple(maximum[index] - minimum[index] for index in range(3))


def bounds_half_extents(bounds: Mapping[str, Any]) -> Vector3:
    extents = bounds_extents(bounds)
    return tuple(float(component) / 2.0 for component in extents)


def bounds_to_lists(bounds: Mapping[str, Any]) -> dict[str, list[float]]:
    return {
        "min": vector_to_list(bounds.get("min")),
        "max": vector_to_list(bounds.get("max")),
    }


def face_center(bounds: Mapping[str, Any], face: str) -> Vector3:
    normalized_face = str(face).strip().lower()
    if normalized_face not in FACE_AXES:
        raise ValueError(f"Unsupported face '{face}'")

    center = list(bounds_center(bounds))
    minimum = coerce_vector3(bounds.get("min"), field_name="bounds.min")
    maximum = coerce_vector3(bounds.get("max"), field_name="bounds.max")
    axis_index, direction = FACE_AXES[normalized_face]
    center[axis_index] = maximum[axis_index] if direction > 0.0 else minimum[axis_index]
    return (center[0], center[1], center[2])


def all_face_centers(bounds: Mapping[str, Any]) -> dict[str, Vector3]:
    return {face: face_center(bounds, face) for face in FACE_NAMES}


def face_centers_to_lists(bounds: Mapping[str, Any]) -> dict[str, list[float]]:
    return {face: vector_to_list(center) for face, center in all_face_centers(bounds).items()}


def interval_gap(first_min: float, first_max: float, second_min: float, second_max: float) -> float:
    if first_max < second_min:
        return float(second_min - first_max)
    if second_max < first_min:
        return float(first_min - second_max)
    overlap = min(first_max, second_max) - max(first_min, second_min)
    return -float(overlap)


def bounds_axis_gap(first_bounds: Mapping[str, Any], second_bounds: Mapping[str, Any], *, axis: int) -> float:
    first_minimum = coerce_vector3(first_bounds.get("min"), field_name="first_bounds.min")
    first_maximum = coerce_vector3(first_bounds.get("max"), field_name="first_bounds.max")
    second_minimum = coerce_vector3(second_bounds.get("min"), field_name="second_bounds.min")
    second_maximum = coerce_vector3(second_bounds.get("max"), field_name="second_bounds.max")
    return interval_gap(first_minimum[axis], first_maximum[axis], second_minimum[axis], second_maximum[axis])


def bounds_overlap_on_axes(
    first_bounds: Mapping[str, Any],
    second_bounds: Mapping[str, Any],
    axes: Iterable[int],
    *,
    tolerance: float = CONTACT_TOLERANCE,
) -> bool:
    return all(bounds_axis_gap(first_bounds, second_bounds, axis=axis) <= float(tolerance) for axis in axes)


def face_gap(
    source_bounds: Mapping[str, Any],
    target_bounds: Mapping[str, Any],
    *,
    source_face: str,
    target_face: str,
) -> float:
    source_face_name = str(source_face).strip().lower()
    target_face_name = str(target_face).strip().lower()
    if source_face_name not in FACE_AXES:
        raise ValueError(f"Unsupported source face '{source_face}'")
    if target_face_name not in FACE_AXES:
        raise ValueError(f"Unsupported target face '{target_face}'")
    source_axis, _ = FACE_AXES[source_face_name]
    target_axis, _ = FACE_AXES[target_face_name]
    if source_axis != target_axis:
        raise ValueError("Face gap requires faces on the same axis")
    source_center = face_center(source_bounds, source_face_name)
    target_center = face_center(target_bounds, target_face_name)
    return abs(float(source_center[source_axis]) - float(target_center[target_axis]))


def lowest_z(obj: Any) -> float:
    return float(world_bounds(obj)["min"][2])


def center_for_lowest_z(bounds: Mapping[str, Any], target_lowest_z: float) -> Vector3:
    center = list(bounds_center(bounds))
    center[2] = float(target_lowest_z) + bounds_half_extents(bounds)[2]
    return (center[0], center[1], center[2])


def is_floor_like(obj: Any) -> bool:
    if object_type(obj) != "MESH":
        return False
    if "floor" in name_of(obj).lower():
        return True
    extents = bounds_extents(world_bounds(obj))
    return extents[2] <= 0.25 and extents[0] >= 2.0 and extents[1] >= 2.0


def is_wall_like(obj: Any) -> bool:
    if object_type(obj) != "MESH":
        return False
    extents = bounds_extents(world_bounds(obj))
    return extents[2] >= 1.0 and min(extents[0], extents[1]) <= 0.35 and max(extents[0], extents[1]) >= 1.0


def is_grounded_on(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    tolerance: float = CONTACT_TOLERANCE,
) -> bool:
    target_minimum = coerce_vector3(target_bounds.get("min"), field_name="target_bounds.min")
    reference_maximum = coerce_vector3(reference_bounds.get("max"), field_name="reference_bounds.max")
    return (
        abs(float(target_minimum[2]) - float(reference_maximum[2])) <= float(tolerance)
        and bounds_overlap_on_axes(target_bounds, reference_bounds, (0, 1), tolerance=tolerance)
    )


def floor_contact_record(
    target: Mapping[str, Any],
    floor_candidates: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = CONTACT_TOLERANCE,
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
        contacts.append({"object": floor_name, "gap": face_gap(target_bounds, floor_bounds, source_face="bottom", target_face="top")})
    if not contacts:
        return None
    return sorted(contacts, key=lambda item: (item["gap"], item["object"]))[0]


def bounds_intersection_extents(first_bounds: Mapping[str, Any], second_bounds: Mapping[str, Any]) -> Vector3:
    first_minimum = coerce_vector3(first_bounds.get("min"), field_name="first_bounds.min")
    first_maximum = coerce_vector3(first_bounds.get("max"), field_name="first_bounds.max")
    second_minimum = coerce_vector3(second_bounds.get("min"), field_name="second_bounds.min")
    second_maximum = coerce_vector3(second_bounds.get("max"), field_name="second_bounds.max")
    return tuple(
        max(0.0, min(first_maximum[index], second_maximum[index]) - max(first_minimum[index], second_minimum[index]))
        for index in range(3)
    )


def bounds_intersection_volume(first_bounds: Mapping[str, Any], second_bounds: Mapping[str, Any]) -> float:
    extents = bounds_intersection_extents(first_bounds, second_bounds)
    return float(extents[0] * extents[1] * extents[2])


def bounds_contains_xy(
    container_bounds: Mapping[str, Any],
    target_bounds: Mapping[str, Any],
    *,
    tolerance: float = CONTACT_TOLERANCE,
) -> bool:
    container_minimum = coerce_vector3(container_bounds.get("min"), field_name="container_bounds.min")
    container_maximum = coerce_vector3(container_bounds.get("max"), field_name="container_bounds.max")
    target_minimum = coerce_vector3(target_bounds.get("min"), field_name="target_bounds.min")
    target_maximum = coerce_vector3(target_bounds.get("max"), field_name="target_bounds.max")
    return (
        target_minimum[0] >= container_minimum[0] - tolerance
        and target_maximum[0] <= container_maximum[0] + tolerance
        and target_minimum[1] >= container_minimum[1] - tolerance
        and target_maximum[1] <= container_maximum[1] + tolerance
    )


def bounds_overlap_area_xy(first_bounds: Mapping[str, Any], second_bounds: Mapping[str, Any]) -> float:
    extents = bounds_intersection_extents(first_bounds, second_bounds)
    return float(extents[0] * extents[1])


def _bounds_volume(bounds: Mapping[str, Any]) -> float:
    extents = bounds_extents(bounds)
    return max(float(extents[0]), 0.0) * max(float(extents[1]), 0.0) * max(float(extents[2]), 0.0)


def _support_candidates_for_object(
    target: Mapping[str, Any],
    objects: Iterable[Mapping[str, Any]],
    *,
    max_gap: float = NEAR_TOLERANCE,
) -> list[dict[str, Any]]:
    target_name = name_of(target)
    target_bounds = world_bounds(target)
    target_bottom = target_bounds["min"][2]
    candidates: list[dict[str, Any]] = []
    for candidate in objects:
        candidate_name = name_of(candidate)
        if not candidate_name or candidate_name == target_name or object_type(candidate) != "MESH":
            continue
        candidate_bounds = world_bounds(candidate)
        candidate_top = candidate_bounds["max"][2]
        gap = float(target_bottom - candidate_top)
        if gap < -CONTACT_TOLERANCE or gap > max_gap:
            continue
        overlap_area = bounds_overlap_area_xy(target_bounds, candidate_bounds)
        if overlap_area <= 0.0:
            continue
        candidates.append(
            {
                "object": candidate_name,
                "gap": gap,
                "overlap_area": overlap_area,
                "contains_xy": bounds_contains_xy(candidate_bounds, target_bounds),
                "is_floor_like": is_floor_like(candidate),
            }
        )
    return sorted(candidates, key=lambda item: (abs(float(item["gap"])), -float(item["overlap_area"]), item["object"]))


def _deepest_intersection(
    target: Mapping[str, Any],
    objects: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    target_name = name_of(target)
    target_bounds = world_bounds(target)
    target_volume = max(_bounds_volume(target_bounds), 0.0001)
    intersections: list[dict[str, Any]] = []
    for candidate in objects:
        candidate_name = name_of(candidate)
        if not candidate_name or candidate_name == target_name or object_type(candidate) != "MESH":
            continue
        candidate_bounds = world_bounds(candidate)
        extents = bounds_intersection_extents(target_bounds, candidate_bounds)
        if not all(component > CONTACT_TOLERANCE for component in extents):
            continue
        volume = float(extents[0] * extents[1] * extents[2])
        candidate_volume = max(_bounds_volume(candidate_bounds), 0.0001)
        smaller_volume = min(target_volume, candidate_volume)
        if volume / smaller_volume < 0.02:
            continue
        intersections.append(
            {
                "object": candidate_name,
                "overlap_extents": vector_to_list(extents),
                "overlap_volume": volume,
                "overlap_ratio": volume / smaller_volume,
            }
        )
    if not intersections:
        return None
    return sorted(intersections, key=lambda item: (-float(item["overlap_volume"]), item["object"]))[0]


def _nearest_clearance(
    target: Mapping[str, Any],
    objects: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    target_name = name_of(target)
    target_bounds = world_bounds(target)
    clearances: list[dict[str, Any]] = []
    for candidate in objects:
        candidate_name = name_of(candidate)
        if not candidate_name or candidate_name == target_name or object_type(candidate) != "MESH":
            continue
        candidate_bounds = world_bounds(candidate)
        z_gap = bounds_axis_gap(target_bounds, candidate_bounds, axis=2)
        if z_gap > CONTACT_TOLERANCE:
            continue
        x_gap = bounds_axis_gap(target_bounds, candidate_bounds, axis=0)
        y_gap = bounds_axis_gap(target_bounds, candidate_bounds, axis=1)
        side_gaps = [gap for gap in (x_gap, y_gap) if gap >= 0.0]
        if not side_gaps:
            continue
        gap = min(side_gaps)
        if gap <= CONTACT_TOLERANCE:
            continue
        clearances.append({"object": candidate_name, "gap": gap})
    if not clearances:
        return None
    return sorted(clearances, key=lambda item: (float(item["gap"]), item["object"]))[0]


def spatial_diagnostics_for_object(
    obj: Mapping[str, Any],
    objects: Iterable[Mapping[str, Any]],
    *,
    scene_bounds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = [record for record in objects if isinstance(record, Mapping)]
    issues: list[str] = []
    suggested_repairs: list[dict[str, Any]] = []
    nearest_support: dict[str, Any] | None = None
    deepest_intersection: dict[str, Any] | None = None
    nearest_clearance: dict[str, Any] | None = None
    target_name = name_of(obj)

    if object_type(obj) != "MESH" or is_floor_like(obj) or is_wall_like(obj):
        return {
            "issues": issues,
            "severity": 0,
            "nearest_support": nearest_support,
            "deepest_intersection": deepest_intersection,
            "nearest_clearance": nearest_clearance,
            "suggested_repairs": suggested_repairs,
        }

    target_bounds = world_bounds(obj)
    supports = _support_candidates_for_object(obj, records)
    contact_supports = [support for support in supports if abs(float(support["gap"])) <= CONTACT_TOLERANCE]
    nearest_support = supports[0] if supports else None

    if target_bounds["min"][2] > CONTACT_TOLERANCE and not contact_supports:
        issues.extend(["floating", "unsupported"])
        repair_params: dict[str, Any] = {"target": target_name}
        if nearest_support is not None:
            repair_params["support"] = nearest_support["object"]
        suggested_repairs.append({"tool": "object.snap_to_support", "params": repair_params})

    floor_supports = [support for support in supports if bool(support.get("is_floor_like"))]
    nearby_floors: list[dict[str, Any]] = list(floor_supports)
    for candidate in records:
        candidate_name = name_of(candidate)
        if not candidate_name or candidate_name == target_name or not is_floor_like(candidate):
            continue
        candidate_bounds = world_bounds(candidate)
        gap = float(target_bounds["min"][2] - candidate_bounds["max"][2])
        if abs(gap) > NEAR_TOLERANCE:
            continue
        if any(item["object"] == candidate_name for item in nearby_floors):
            continue
        nearby_floors.append(
            {
                "object": candidate_name,
                "gap": gap,
                "overlap_area": bounds_overlap_area_xy(target_bounds, candidate_bounds),
                "contains_xy": bounds_contains_xy(candidate_bounds, target_bounds),
                "is_floor_like": True,
            }
        )
    nearby_floors = sorted(nearby_floors, key=lambda item: (abs(float(item["gap"])), -float(item["overlap_area"]), item["object"]))
    if nearby_floors and not any(bool(support.get("contains_xy")) for support in nearby_floors):
        issues.append("outside_support")
        suggested_repairs.append(
            {
                "tool": "object.fit_inside",
                "params": {"target": target_name, "container": nearby_floors[0]["object"], "align_bottom": True},
            }
        )

    deepest_intersection = _deepest_intersection(obj, records)
    if deepest_intersection is not None:
        issues.append("deep_intersection")
        suggested_repairs.append(
            {
                "tool": "object.resolve_overlap",
                "params": {"target": target_name, "obstacle": deepest_intersection["object"]},
            }
        )

    nearest_clearance = _nearest_clearance(obj, records)
    if nearest_clearance is not None and float(nearest_clearance["gap"]) <= 0.1:
        issues.append("crowded")

    if scene_bounds is not None and len(records) >= 3:
        scene_extents = bounds_extents(scene_bounds)
        target_extents = bounds_extents(target_bounds)
        if (
            scene_extents[0] > 0.0
            and scene_extents[1] > 0.0
            and scene_extents[2] > 0.0
            and target_extents[0] / scene_extents[0] >= 0.65
            and target_extents[1] / scene_extents[1] >= 0.65
            and target_extents[2] / scene_extents[2] >= 0.55
        ):
            issues.append("oversized_blocking_form")

    issue_weights = {
        "deep_intersection": 3,
        "floating": 2,
        "outside_support": 2,
        "unsupported": 2,
        "oversized_blocking_form": 2,
        "crowded": 1,
    }
    return {
        "issues": _dedupe_relations(issues),
        "severity": sum(issue_weights.get(issue, 1) for issue in _dedupe_relations(issues)),
        "nearest_support": nearest_support,
        "deepest_intersection": deepest_intersection,
        "nearest_clearance": nearest_clearance,
        "suggested_repairs": suggested_repairs,
    }


def spatial_diagnostics(objects: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = sorted(
        [obj for obj in objects if isinstance(obj, Mapping) and isinstance(obj.get("name"), str)],
        key=lambda item: str(item.get("name", "")),
    )
    scene_bounds = _scene_bounds_from_records(records)
    by_object: dict[str, dict[str, Any]] = {}
    top_issues: list[dict[str, Any]] = []
    for obj in records:
        name = str(obj.get("name", "")).strip()
        if not name:
            continue
        diagnostics = spatial_diagnostics_for_object(obj, records, scene_bounds=scene_bounds)
        by_object[name] = diagnostics
        for issue in diagnostics.get("issues", []):
            top_issues.append(
                {
                    "object": name,
                    "issue": issue,
                    "severity": diagnostics.get("severity", 0),
                    "suggested_repairs": diagnostics.get("suggested_repairs", []),
                }
            )
    top_issues = sorted(top_issues, key=lambda item: (-int(item.get("severity", 0)), item["object"], item["issue"]))
    return {
        "issue_count": len(top_issues),
        "top_issues": top_issues[:3],
        "by_object": by_object,
    }


def place_on_surface_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    surface: str = "top",
    offset: float = 0.0,
) -> Vector3:
    normalized_surface = str(surface).strip().lower()
    if normalized_surface not in FACE_AXES:
        raise ValueError(f"Unsupported surface '{surface}'")
    reference_center = list(face_center(reference_bounds, normalized_surface))
    target_half_extents = bounds_half_extents(target_bounds)
    axis_index, direction = FACE_AXES[normalized_surface]
    reference_center[axis_index] += direction * (target_half_extents[axis_index] + float(offset))
    return (reference_center[0], reference_center[1], reference_center[2])


def place_against_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    side: str,
    offset: float = 0.0,
    current_location: Any = None,
) -> Vector3:
    location = list(place_on_surface_location(target_bounds, reference_bounds, surface=side, offset=offset))
    if current_location is None:
        return (location[0], location[1], location[2])
    axis_index, _ = FACE_AXES[str(side).strip().lower()]
    current = coerce_vector3(current_location, field_name="current_location")
    for index in range(3):
        if index != axis_index:
            location[index] = current[index]
    return (location[0], location[1], location[2])


def place_relative_location(
    target_bounds: Mapping[str, Any],
    reference_bounds: Mapping[str, Any],
    *,
    relation: str,
    distance: float,
    current_location: Any = None,
) -> Vector3:
    normalized_relation = str(relation).strip().lower()
    if normalized_relation not in RELATION_TO_FACE:
        raise ValueError(f"Unsupported relation '{relation}'")
    location = list(
        place_on_surface_location(
            target_bounds,
            reference_bounds,
            surface=RELATION_TO_FACE[normalized_relation],
            offset=float(distance),
        )
    )
    if current_location is not None and normalized_relation in {"left_of", "right_of", "next_to", "behind", "in_front_of"}:
        current = coerce_vector3(current_location, field_name="current_location")
        location[2] = current[2]
    return (location[0], location[1], location[2])


def align_to_location(target_bounds: Mapping[str, Any], reference_bounds: Mapping[str, Any], *, axis: str) -> Vector3:
    normalized_axis = str(axis).strip().lower()
    if normalized_axis not in AXIS_NAMES:
        raise ValueError(f"Unsupported axis '{axis}'")
    axis_index = AXIS_NAMES[normalized_axis]
    location = list(bounds_center(target_bounds))
    reference_center = bounds_center(reference_bounds)
    location[axis_index] = reference_center[axis_index]
    return (location[0], location[1], location[2])


def classify_spatial_relation(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    contact_tolerance: float = CONTACT_TOLERANCE,
    near_tolerance: float = NEAR_TOLERANCE,
) -> list[str]:
    source_bounds = world_bounds(source)
    target_bounds = world_bounds(target)
    source_center = bounds_center(source_bounds)
    target_center = bounds_center(target_bounds)
    source_floor_like = is_floor_like(source)
    target_floor_like = is_floor_like(target)
    relations: list[str] = []

    xy_overlap = bounds_overlap_on_axes(source_bounds, target_bounds, (0, 1), tolerance=contact_tolerance)
    xz_overlap = bounds_overlap_on_axes(source_bounds, target_bounds, (0, 2), tolerance=contact_tolerance)
    yz_overlap = bounds_overlap_on_axes(source_bounds, target_bounds, (1, 2), tolerance=contact_tolerance)

    if is_grounded_on(source_bounds, target_bounds, tolerance=contact_tolerance):
        relations.append("on")

    if xy_overlap:
        z_gap = bounds_axis_gap(source_bounds, target_bounds, axis=2)
        if -contact_tolerance <= z_gap <= near_tolerance:
            relations.append("above" if source_center[2] > target_center[2] else "below")

    if source_floor_like or target_floor_like:
        return _dedupe_relations(relations)

    x_gap = bounds_axis_gap(source_bounds, target_bounds, axis=0)
    if yz_overlap and -contact_tolerance <= x_gap <= near_tolerance:
        if abs(x_gap) <= contact_tolerance:
            relations.append("against")
        relations.append("left_of" if source_center[0] < target_center[0] else "right_of")
        relations.append("next_to")

    y_gap = bounds_axis_gap(source_bounds, target_bounds, axis=1)
    if xz_overlap and -contact_tolerance <= y_gap <= near_tolerance:
        if abs(y_gap) <= contact_tolerance:
            relations.append("against")
        relations.append("behind" if source_center[1] < target_center[1] else "in_front_of")
        relations.append("next_to")

    return _dedupe_relations(relations)


def _dedupe_relations(relations: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for relation in relations:
        if relation not in deduped:
            deduped.append(relation)
    return deduped


def spatial_relations(
    objects: Iterable[Mapping[str, Any]],
    *,
    contact_tolerance: float = CONTACT_TOLERANCE,
    near_tolerance: float = NEAR_TOLERANCE,
) -> list[dict[str, Any]]:
    records = sorted(
        [obj for obj in objects if isinstance(obj.get("name"), str)],
        key=lambda item: str(item.get("name", "")),
    )
    relations: list[dict[str, Any]] = []
    for source in records:
        source_name = str(source.get("name", "")).strip()
        if not source_name:
            continue
        for target in records:
            target_name = str(target.get("name", "")).strip()
            if not target_name or source_name == target_name:
                continue
            for relation in classify_spatial_relation(
                source,
                target,
                contact_tolerance=contact_tolerance,
                near_tolerance=near_tolerance,
            ):
                relations.append({"source": source_name, "target": target_name, "relation": relation})
    return sorted(relations, key=lambda item: (item["source"], item["target"], item["relation"]))


def spatial_metadata_for_object(
    obj: Mapping[str, Any],
    *,
    floor_candidates: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    bounds = world_bounds(obj)
    floor_contact = floor_contact_record(obj, floor_candidates)
    return {
        "world_bounds": bounds_to_lists(bounds),
        "center": vector_to_list(bounds_center(bounds)),
        "extents": vector_to_list(bounds_extents(bounds)),
        "half_extents": vector_to_list(bounds_half_extents(bounds)),
        "face_centers": face_centers_to_lists(bounds),
        "grounded": floor_contact is not None or abs(float(bounds["min"][2])) <= CONTACT_TOLERANCE,
        "floor_contact": floor_contact,
        "is_floor_like": is_floor_like(obj),
        "is_wall_like": is_wall_like(obj),
    }


def _scene_bounds_from_records(records: list[Mapping[str, Any]]) -> Bounds | None:
    if not records:
        return None
    bounds = [world_bounds(record) for record in records]
    return {
        "min": tuple(min(float(item["min"][index]) for item in bounds) for index in range(3)),
        "max": tuple(max(float(item["max"][index]) for item in bounds) for index in range(3)),
    }


def _wall_inner_face(wall_bounds: Mapping[str, Any], scene_bounds: Mapping[str, Any]) -> str | None:
    extents = bounds_extents(wall_bounds)
    thin_axis = 0 if extents[0] <= extents[1] else 1
    if extents[thin_axis] > 0.35:
        return None
    wall_center = bounds_center(wall_bounds)
    scene_center = bounds_center(scene_bounds)
    if thin_axis == 0:
        return "right" if wall_center[0] <= scene_center[0] else "left"
    return "front" if wall_center[1] <= scene_center[1] else "back"


def _floor_corner_anchors(name: str, bounds: Mapping[str, Any]) -> list[dict[str, Any]]:
    minimum = coerce_vector3(bounds.get("min"), field_name="bounds.min")
    maximum = coerce_vector3(bounds.get("max"), field_name="bounds.max")
    z_value = maximum[2]
    anchors: list[dict[str, Any]] = []
    for x_name, x_value in (("left", minimum[0]), ("right", maximum[0])):
        for y_name, y_value in (("back", minimum[1]), ("front", maximum[1])):
            anchors.append(
                {
                    "name": f"{name}.floor.corner.{x_name}.{y_name}",
                    "type": "floor_corner",
                    "object": name,
                    "corner": f"{x_name}.{y_name}",
                    "location": [float(x_value), float(y_value), float(z_value)],
                }
            )
    return anchors


def spatial_anchors(objects: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = sorted(
        [obj for obj in objects if isinstance(obj.get("name"), str)],
        key=lambda item: str(item.get("name", "")),
    )
    scene_bounds = _scene_bounds_from_records(records)
    anchors: list[dict[str, Any]] = []

    for obj in records:
        name = str(obj.get("name", "")).strip()
        if not name:
            continue
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
            anchors.extend(
                [
                    {
                        "name": f"{name}.floor.top",
                        "type": "floor_top",
                        "object": name,
                        "location": vector_to_list(face_center(bounds, "top")),
                    },
                    {
                        "name": f"{name}.floor.center",
                        "type": "floor_center",
                        "object": name,
                        "location": vector_to_list(bounds_center(bounds)),
                    },
                ]
            )
            anchors.extend(_floor_corner_anchors(name, bounds))

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
        minimum = coerce_vector3(scene_bounds.get("min"), field_name="scene_bounds.min")
        maximum = coerce_vector3(scene_bounds.get("max"), field_name="scene_bounds.max")
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
        if "floor" in name_of(obj).lower():
            named_candidates.append(obj)
        else:
            flat_candidates.append(obj)
    if named_candidates:
        return sorted(named_candidates, key=lambda candidate: name_of(candidate).lower())[0]
    if flat_candidates:
        return sorted(flat_candidates, key=lambda candidate: (lowest_z(candidate), name_of(candidate).lower()))[0]
    return None
