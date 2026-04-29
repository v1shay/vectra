from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spatial import (
    CONTACT_TOLERANCE,
    bounds_center,
    bounds_extents,
    bounds_half_extents,
    bounds_intersection_extents,
    bounds_to_lists,
    face_center,
    is_floor_like,
    name_of,
    object_type,
    place_against_location,
    place_on_surface_location,
    spatial_diagnostics,
    world_bounds,
)


SEVERE_SPATIAL_ISSUES = {
    "deep_intersection",
    "floating",
    "outside_support",
    "unsupported",
}


@dataclass(frozen=True)
class SpatialConstraint:
    kind: str
    target: str
    reference: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialSolution:
    target: str
    location: tuple[float, float, float]
    constraint: SpatialConstraint
    reason: str


@dataclass(frozen=True)
class SpatialValidationReport:
    ok: bool
    issue_count: int
    top_issues: list[dict[str, Any]]
    repair_actions: list[dict[str, Any]]


def object_records_from_scene_objects(objects: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in objects:
        object_name = name_of(obj)
        if not object_name:
            continue
        records.append(
            {
                "name": object_name,
                "type": object_type(obj),
                "location": _vector_list(getattr(obj, "location", (0.0, 0.0, 0.0))),
                "dimensions": _vector_list(getattr(obj, "dimensions", (2.0, 2.0, 2.0))),
                "bounds": bounds_to_lists(world_bounds(obj)),
            }
        )
    return records


def constraints_for_tool_action(tool_name: str, params: dict[str, Any]) -> list[SpatialConstraint]:
    target = _string_value(params.get("target"))
    reference = _string_value(params.get("reference") or params.get("container") or params.get("support"))
    if not target:
        return []
    if tool_name == "object.place_on_surface":
        return [
            SpatialConstraint(
                kind="support" if str(params.get("surface", "top")).strip().lower() == "top" else "face_contact",
                target=target,
                reference=reference,
                params={"surface": params.get("surface", "top"), "offset": params.get("offset", 0.0)},
            )
        ]
    if tool_name == "object.place_against":
        return [
            SpatialConstraint(
                kind="against",
                target=target,
                reference=reference,
                params={"side": params.get("side"), "offset": params.get("offset", 0.0)},
            )
        ]
    if tool_name == "object.fit_inside":
        return [SpatialConstraint(kind="inside", target=target, reference=reference, params=dict(params))]
    if tool_name == "object.resolve_overlap":
        return [SpatialConstraint(kind="no_overlap", target=target, reference=reference, params=dict(params))]
    if tool_name == "object.snap_to_support":
        return [SpatialConstraint(kind="support", target=target, reference=reference, params=dict(params))]
    return []


class SpatialConstraintSolver:
    def __init__(self, objects: list[Any]) -> None:
        self.objects = [obj for obj in objects if name_of(obj)]
        self.records = object_records_from_scene_objects(self.objects)

    def solve(self, constraint: SpatialConstraint) -> SpatialSolution:
        target = self._object(constraint.target)
        if constraint.kind == "support":
            support = self._object(constraint.reference) if constraint.reference else self._best_support(target)
            location = place_on_surface_location(world_bounds(target), world_bounds(support), surface="top", offset=0.0)
            location = _add_vector(location, constraint.params.get("offset_vector"))
            return SpatialSolution(
                target=name_of(target),
                location=location,
                constraint=constraint,
                reason=f"Placed '{name_of(target)}' on support '{name_of(support)}'.",
            )
        if constraint.kind == "against":
            reference = self._object(constraint.reference)
            side = str(constraint.params.get("side", "front")).strip().lower()
            location = place_against_location(
                world_bounds(target),
                world_bounds(reference),
                side=side,
                offset=float(constraint.params.get("offset", 0.0) or 0.0),
                current_location=getattr(target, "location", None),
            )
            location = _add_vector(location, constraint.params.get("offset_vector"))
            return SpatialSolution(
                target=name_of(target),
                location=location,
                constraint=constraint,
                reason=f"Placed '{name_of(target)}' against '{name_of(reference)}'.",
            )
        if constraint.kind == "inside":
            container = self._object(constraint.reference)
            location = _fit_inside_location(target, container, align_bottom=bool(constraint.params.get("align_bottom", False)))
            return SpatialSolution(
                target=name_of(target),
                location=location,
                constraint=constraint,
                reason=f"Fit '{name_of(target)}' inside '{name_of(container)}'.",
            )
        if constraint.kind == "no_overlap":
            obstacle = self._object(constraint.reference) if constraint.reference else self._deepest_overlap(target)
            location = _separate_location(target, obstacle, padding=float(constraint.params.get("padding", 0.02) or 0.02))
            return SpatialSolution(
                target=name_of(target),
                location=location,
                constraint=constraint,
                reason=f"Separated '{name_of(target)}' from '{name_of(obstacle)}'.",
            )
        if constraint.kind == "face_visibility":
            reference = self._object(constraint.reference)
            face = str(constraint.params.get("face", "front")).strip().lower()
            face_location = face_center(world_bounds(reference), face)
            return SpatialSolution(
                target=name_of(target),
                location=face_location,
                constraint=constraint,
                reason=f"Aligned '{name_of(target)}' to visible face '{face}' of '{name_of(reference)}'.",
            )
        raise ValueError(f"Unsupported spatial constraint kind '{constraint.kind}'")

    def validate(self, *, affected_names: list[str] | None = None) -> SpatialValidationReport:
        return validate_spatial_records(self.records, affected_names=affected_names)

    def _object(self, raw_name: str | None) -> Any:
        if raw_name is None:
            raise ValueError("Spatial constraint requires a reference object")
        exact = [obj for obj in self.objects if name_of(obj) == raw_name]
        if exact:
            return exact[0]
        lowered = raw_name.lower()
        fuzzy = [obj for obj in self.objects if name_of(obj).lower() == lowered or lowered in name_of(obj).lower()]
        if len(fuzzy) == 1:
            return fuzzy[0]
        if len(fuzzy) > 1:
            names = sorted(name_of(obj) for obj in fuzzy)
            raise ValueError(f"Spatial constraint reference '{raw_name}' matched multiple objects: {names}")
        raise ValueError(f"Spatial constraint reference '{raw_name}' could not be resolved")

    def _best_support(self, target: Any) -> Any:
        target_bounds = world_bounds(target)
        scored: list[tuple[float, float, str, Any]] = []
        for candidate in self.objects:
            if name_of(candidate) == name_of(target) or object_type(candidate) != "MESH":
                continue
            candidate_bounds = world_bounds(candidate)
            gap = float(target_bounds["min"][2] - candidate_bounds["max"][2])
            if gap < -CONTACT_TOLERANCE or gap > 5.0:
                continue
            overlap = bounds_intersection_extents(target_bounds, candidate_bounds)
            overlap_area = float(overlap[0] * overlap[1])
            if overlap_area <= 0.0:
                continue
            floor_bonus = -0.25 if is_floor_like(candidate) else 0.0
            scored.append((abs(gap) + floor_bonus, -overlap_area, name_of(candidate), candidate))
        if not scored:
            raise ValueError(f"No support could be inferred for '{name_of(target)}'")
        return sorted(scored)[0][3]

    def _deepest_overlap(self, target: Any) -> Any:
        target_bounds = world_bounds(target)
        scored: list[tuple[float, str, Any]] = []
        for candidate in self.objects:
            if name_of(candidate) == name_of(target) or object_type(candidate) != "MESH":
                continue
            overlap = bounds_intersection_extents(target_bounds, world_bounds(candidate))
            volume = float(overlap[0] * overlap[1] * overlap[2])
            if volume > 0.0:
                scored.append((-volume, name_of(candidate), candidate))
        if not scored:
            raise ValueError(f"No overlap could be inferred for '{name_of(target)}'")
        return sorted(scored)[0][2]


def validate_spatial_records(
    records: list[dict[str, Any]],
    *,
    affected_names: list[str] | None = None,
) -> SpatialValidationReport:
    diagnostics = spatial_diagnostics(records)
    affected = set(affected_names or [])
    repair_actions: list[dict[str, Any]] = []
    top_issues: list[dict[str, Any]] = []
    by_object = diagnostics.get("by_object", {})
    if isinstance(by_object, dict):
        for object_name, object_diagnostics in by_object.items():
            if affected and object_name not in affected:
                continue
            if not isinstance(object_diagnostics, dict):
                continue
            issues = object_diagnostics.get("issues", [])
            if not isinstance(issues, list):
                continue
            severe = [issue for issue in issues if issue in SEVERE_SPATIAL_ISSUES]
            if not severe:
                continue
            top_issues.append(
                {
                    "object": object_name,
                    "issues": severe,
                    "severity": object_diagnostics.get("severity", 0),
                }
            )
            suggested = object_diagnostics.get("suggested_repairs", [])
            if isinstance(suggested, list):
                repair_actions.extend(action for action in suggested if isinstance(action, dict))
    top_issues = sorted(top_issues, key=lambda item: (-int(item.get("severity", 0)), item["object"]))
    return SpatialValidationReport(
        ok=not top_issues,
        issue_count=len(top_issues),
        top_issues=top_issues[:3],
        repair_actions=repair_actions[:6],
    )


def _fit_inside_location(target: Any, container: Any, *, align_bottom: bool = False) -> tuple[float, float, float]:
    target_bounds = world_bounds(target)
    container_bounds = world_bounds(container)
    half_extents = bounds_half_extents(target_bounds)
    location = list(_vector_tuple(getattr(target, "location", bounds_center(target_bounds))))
    for axis in (0, 1):
        minimum = float(container_bounds["min"][axis]) + half_extents[axis]
        maximum = float(container_bounds["max"][axis]) - half_extents[axis]
        location[axis] = (float(container_bounds["min"][axis]) + float(container_bounds["max"][axis])) / 2.0 if minimum > maximum else min(max(location[axis], minimum), maximum)
    if align_bottom or is_floor_like(container):
        location[2] = float(container_bounds["max"][2]) + half_extents[2]
    return (location[0], location[1], location[2])


def _separate_location(target: Any, obstacle: Any, *, padding: float) -> tuple[float, float, float]:
    target_bounds = world_bounds(target)
    obstacle_bounds = world_bounds(obstacle)
    overlap = bounds_intersection_extents(target_bounds, obstacle_bounds)
    if not all(component > 0.0 for component in overlap):
        return _vector_tuple(getattr(target, "location", bounds_center(target_bounds)))
    axis = min(range(3), key=lambda index: overlap[index])
    target_center = bounds_center(target_bounds)
    obstacle_center = bounds_center(obstacle_bounds)
    direction = -1.0 if target_center[axis] < obstacle_center[axis] else 1.0
    location = list(_vector_tuple(getattr(target, "location", target_center)))
    location[axis] += direction * (float(overlap[axis]) + float(padding))
    return (location[0], location[1], location[2])


def _add_vector(location: tuple[float, float, float], raw_offset: Any) -> tuple[float, float, float]:
    offset = _vector_tuple(raw_offset) if raw_offset is not None else (0.0, 0.0, 0.0)
    return (
        float(location[0]) + offset[0],
        float(location[1]) + offset[1],
        float(location[2]) + offset[2],
    )


def _vector_list(value: Any) -> list[float]:
    vector = _vector_tuple(value)
    return [vector[0], vector[1], vector[2]]


def _vector_tuple(value: Any) -> tuple[float, float, float]:
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError, KeyError):
        return (0.0, 0.0, 0.0)


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
