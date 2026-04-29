from __future__ import annotations

import math
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import ensure_object_mode, normalize_optional_string, scene_objects, validate_vector3
from .registry import register_tool
from .spatial import (
    CONTACT_TOLERANCE,
    align_to_location,
    bounds_center,
    bounds_contains_xy,
    bounds_extents,
    bounds_half_extents,
    bounds_intersection_extents,
    bounds_to_lists,
    is_floor_like,
    object_type,
    place_against_location,
    place_on_surface_location,
    place_relative_location,
    spatial_anchors,
    world_bounds,
)

_SUPPORTED_SURFACES = {"top", "bottom", "left", "right", "front", "back"}
_SUPPORTED_RELATIONS = {"above", "below", "behind", "in_front_of", "left_of", "next_to", "right_of"}
_SUPPORTED_AXES = {"x", "y", "z"}
_ZERO_OFFSET = (0.0, 0.0, 0.0)


def _normalize_distance(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ToolValidationError(f"'{field_name}' must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"'{field_name}' must be numeric") from exc
    if not math.isfinite(normalized):
        raise ToolValidationError(f"'{field_name}' must be finite")
    if normalized < 0.0:
        raise ToolValidationError(f"'{field_name}' must be greater than or equal to 0")
    return normalized


def _normalize_offset_vector(value: Any, field_name: str) -> tuple[float, float, float]:
    if value is None:
        return _ZERO_OFFSET
    return validate_vector3(value, field_name)


def _required_object_name(value: Any, field_name: str) -> str:
    normalized = normalize_optional_string(value, field_name)
    if normalized is None:
        raise ToolValidationError(f"'{field_name}' must be a non-empty string")
    return normalized


def _resolved_bounds_payload(obj: Any) -> dict[str, list[float]]:
    return bounds_to_lists(world_bounds(obj))


def _update_view_layer(context: Any) -> None:
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None and bpy is not None:
        view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        update()


def _placement_result(
    *,
    target: Any,
    reference_name: str,
    location: tuple[float, float, float],
    placement_mode: str,
    placement_reason: str,
    anchor_name: str | None = None,
) -> ToolExecutionResult:
    outputs = {
        "object_name": target.name,
        "object_names": [target.name],
        "placement_mode": placement_mode,
        "placement_reason": placement_reason,
        "reference_object": reference_name,
        "resolved_location": [float(component) for component in location],
        "resolved_bounds": _resolved_bounds_payload(target),
    }
    if anchor_name is not None:
        outputs["anchor"] = anchor_name
    return ToolExecutionResult(
        outputs=outputs,
        message=f"Placed '{target.name}' using {placement_mode}",
    )


def _resolve_object_strict(context: Any, object_name: str, field_name: str) -> Any:
    candidates = [
        obj
        for obj in scene_objects(context)
        if isinstance(getattr(obj, "name", None), str)
    ]
    exact = [obj for obj in candidates if obj.name == object_name]
    if exact:
        return exact[0]

    lowered = object_name.lower()
    fuzzy = [
        obj
        for obj in candidates
        if obj.name.lower() == lowered or lowered in obj.name.lower()
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        names = sorted(obj.name for obj in fuzzy)
        raise ToolExecutionError(f"Spatial field '{field_name}' matched multiple objects: {names}")
    raise ToolExecutionError(f"Spatial field '{field_name}' could not resolve object '{object_name}'")


def _resolve_target_and_reference(context: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    target_name = _required_object_name(params.get("target"), "target")
    reference_name = _required_object_name(params.get("reference"), "reference")
    if target_name == reference_name:
        raise ToolExecutionError("The target and reference objects must be different")
    target = _resolve_object_strict(context, target_name, "target")
    reference = _resolve_object_strict(context, reference_name, "reference")
    if target == reference or getattr(target, "name", None) == getattr(reference, "name", None):
        raise ToolExecutionError("The target and reference objects must be different")
    return target, reference


def _apply_offset_vector(
    location: tuple[float, float, float],
    offset_vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(location[0]) + float(offset_vector[0]),
        float(location[1]) + float(offset_vector[1]),
        float(location[2]) + float(offset_vector[2]),
    )


def _set_object_location(obj: Any, location: tuple[float, float, float]) -> None:
    obj.location = (float(location[0]), float(location[1]), float(location[2]))


def _mesh_objects(context: Any) -> list[Any]:
    return [obj for obj in scene_objects(context) if object_type(obj) == "MESH"]


def _support_score(target: Any, candidate: Any) -> tuple[float, float, str] | None:
    target_bounds = world_bounds(target)
    candidate_bounds = world_bounds(candidate)
    gap = float(target_bounds["min"][2] - candidate_bounds["max"][2])
    if gap < -CONTACT_TOLERANCE or gap > 5.0:
        return None
    overlap_extents = bounds_intersection_extents(target_bounds, candidate_bounds)
    overlap_area = float(overlap_extents[0] * overlap_extents[1])
    if overlap_area <= 0.0:
        return None
    target_extents = bounds_extents(target_bounds)
    target_area = max(float(target_extents[0] * target_extents[1]), 0.0001)
    overlap_ratio = overlap_area / target_area
    floor_bonus = -0.25 if is_floor_like(candidate) else 0.0
    return (abs(gap) + floor_bonus, -overlap_ratio, getattr(candidate, "name", ""))


def _best_support(context: Any, target: Any, support_name: str | None = None) -> Any:
    if support_name:
        support = _resolve_object_strict(context, support_name, "support")
        if support == target or getattr(support, "name", None) == getattr(target, "name", None):
            raise ToolExecutionError("The target and support objects must be different")
        return support

    scored: list[tuple[tuple[float, float, str], Any]] = []
    for candidate in _mesh_objects(context):
        if candidate == target or getattr(candidate, "name", None) == getattr(target, "name", None):
            continue
        score = _support_score(target, candidate)
        if score is not None:
            scored.append((score, candidate))
    if not scored:
        raise ToolExecutionError(f"No support surface could be inferred for '{target.name}'")
    return sorted(scored, key=lambda item: item[0])[0][1]


def _deepest_overlap(context: Any, target: Any, obstacle_name: str | None = None) -> Any:
    if obstacle_name:
        obstacle = _resolve_object_strict(context, obstacle_name, "obstacle")
        if obstacle == target or getattr(obstacle, "name", None) == getattr(target, "name", None):
            raise ToolExecutionError("The target and obstacle objects must be different")
        return obstacle

    target_bounds = world_bounds(target)
    overlaps: list[tuple[float, str, Any]] = []
    for candidate in _mesh_objects(context):
        if candidate == target or getattr(candidate, "name", None) == getattr(target, "name", None):
            continue
        extents = bounds_intersection_extents(target_bounds, world_bounds(candidate))
        if not all(component > CONTACT_TOLERANCE for component in extents):
            continue
        volume = float(extents[0] * extents[1] * extents[2])
        overlaps.append((-volume, getattr(candidate, "name", ""), candidate))
    if not overlaps:
        raise ToolExecutionError(f"No overlapping object could be inferred for '{target.name}'")
    return sorted(overlaps)[0][2]


def _mesh_anchor_records(context: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in scene_objects(context):
        if object_type(obj) != "MESH":
            continue
        bounds = _resolved_bounds_payload(obj)
        records.append(
            {
                "name": obj.name,
                "type": obj.type,
                "bounds": bounds,
                "location": [float(component) for component in getattr(obj, "location", (0.0, 0.0, 0.0))[:3]],
                "dimensions": [float(component) for component in getattr(obj, "dimensions", (2.0, 2.0, 2.0))[:3]],
            }
        )
    return records


def _resolve_anchor(context: Any, anchor_name: str) -> dict[str, Any]:
    anchors = spatial_anchors(_mesh_anchor_records(context))
    exact = [anchor for anchor in anchors if anchor.get("name") == anchor_name]
    if exact:
        return exact[0]
    lowered = anchor_name.lower()
    fuzzy = [
        anchor
        for anchor in anchors
        if isinstance(anchor.get("name"), str)
        and (anchor["name"].lower() == lowered or lowered in anchor["name"].lower())
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        names = sorted(str(anchor.get("name")) for anchor in fuzzy)
        raise ToolExecutionError(f"Spatial anchor '{anchor_name}' matched multiple anchors: {names}")
    raise ToolExecutionError(f"Spatial anchor '{anchor_name}' could not be resolved from current geometry")


def _anchor_location_for_target(
    target: Any,
    anchor: dict[str, Any],
    *,
    align_bottom: bool,
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    raw_location = anchor.get("location")
    if not isinstance(raw_location, list) or len(raw_location) != 3:
        raise ToolExecutionError(f"Spatial anchor '{anchor.get('name')}' has no valid location")
    location = [float(component) for component in raw_location]
    half_extents = bounds_half_extents(world_bounds(target))

    if anchor.get("type") == "floor_corner":
        corner = str(anchor.get("corner", ""))
        if "left" in corner:
            location[0] += half_extents[0]
        elif "right" in corner:
            location[0] -= half_extents[0]
        if "back" in corner:
            location[1] += half_extents[1]
        elif "front" in corner:
            location[1] -= half_extents[1]
        if align_bottom:
            location[2] += half_extents[2]
    elif align_bottom:
        location[2] += half_extents[2]

    return (
        location[0] + float(offset[0]),
        location[1] + float(offset[1]),
        location[2] + float(offset[2]),
    )


@register_tool
class PlaceOnSurfaceTool(BaseTool):
    name = "object.place_on_surface"
    description = "Place one object so it sits in contact with a chosen AABB face of a reference object."
    input_schema = {
        "target": {"type": "string", "required": True},
        "reference": {"type": "string", "required": True},
        "surface": {"type": "string", "required": False, "enum": sorted(_SUPPORTED_SURFACES)},
        "offset": {"type": "number", "required": False},
        "offset_vector": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        reference = _required_object_name(params.get("reference"), "reference")
        surface = str(params.get("surface", "top")).strip().lower() or "top"
        if surface not in _SUPPORTED_SURFACES:
            raise ToolValidationError(f"'surface' must be one of {sorted(_SUPPORTED_SURFACES)}")
        offset = _normalize_distance(params.get("offset", 0.0), "offset")
        offset_vector = _normalize_offset_vector(params.get("offset_vector"), "offset_vector")
        return {
            "target": target,
            "reference": reference,
            "surface": surface,
            "offset": offset,
            "offset_vector": offset_vector,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target, reference = _resolve_target_and_reference(context, validated)
        location = place_on_surface_location(
            world_bounds(target),
            world_bounds(reference),
            surface=validated["surface"],
            offset=float(validated["offset"]),
        )
        location = _apply_offset_vector(location, validated["offset_vector"])
        _set_object_location(target, location)
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=reference.name,
            location=location,
            placement_mode="surface_contact",
            placement_reason=f"Placed '{target.name}' on the {validated['surface']} face of '{reference.name}'.",
        )


@register_tool
class PlaceAgainstTool(BaseTool):
    name = "object.place_against"
    description = "Place one object against a chosen AABB side of a reference object."
    input_schema = {
        "target": {"type": "string", "required": True},
        "reference": {"type": "string", "required": True},
        "side": {"type": "string", "required": True, "enum": sorted(_SUPPORTED_SURFACES)},
        "offset": {"type": "number", "required": False},
        "offset_vector": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        reference = _required_object_name(params.get("reference"), "reference")
        side = _required_object_name(params.get("side"), "side")
        normalized_side = str(side).strip().lower()
        if normalized_side not in _SUPPORTED_SURFACES:
            raise ToolValidationError(f"'side' must be one of {sorted(_SUPPORTED_SURFACES)}")
        offset = _normalize_distance(params.get("offset", 0.0), "offset")
        offset_vector = _normalize_offset_vector(params.get("offset_vector"), "offset_vector")
        return {
            "target": target,
            "reference": reference,
            "side": normalized_side,
            "offset": offset,
            "offset_vector": offset_vector,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target, reference = _resolve_target_and_reference(context, validated)
        location = place_against_location(
            world_bounds(target),
            world_bounds(reference),
            side=validated["side"],
            offset=float(validated["offset"]),
            current_location=getattr(target, "location", None),
        )
        location = _apply_offset_vector(location, validated["offset_vector"])
        _set_object_location(target, location)
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=reference.name,
            location=location,
            placement_mode="against_contact",
            placement_reason=f"Placed '{target.name}' against the {validated['side']} side of '{reference.name}'.",
        )


@register_tool
class SnapToSupportTool(BaseTool):
    name = "object.snap_to_support"
    description = "Snap an object onto a named or best inferred support surface using current geometry."
    input_schema = {
        "target": {"type": "string", "required": True},
        "support": {"type": "string", "required": False},
        "offset_vector": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        support = normalize_optional_string(params.get("support"), "support")
        offset_vector = _normalize_offset_vector(params.get("offset_vector"), "offset_vector")
        return {
            "target": target,
            "support": support,
            "offset_vector": offset_vector,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target = _resolve_object_strict(context, validated["target"], "target")
        support = _best_support(context, target, validated.get("support"))
        location = place_on_surface_location(world_bounds(target), world_bounds(support), surface="top", offset=0.0)
        location = _apply_offset_vector(location, validated["offset_vector"])
        _set_object_location(target, location)
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=support.name,
            location=location,
            placement_mode="support_snap",
            placement_reason=f"Snapped '{target.name}' onto support '{support.name}'.",
        )


@register_tool
class ResolveOverlapTool(BaseTool):
    name = "object.resolve_overlap"
    description = "Move an object along the shallowest AABB separating axis to clear an overlap."
    input_schema = {
        "target": {"type": "string", "required": True},
        "obstacle": {"type": "string", "required": False},
        "padding": {"type": "number", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        obstacle = normalize_optional_string(params.get("obstacle"), "obstacle")
        padding = _normalize_distance(params.get("padding", 0.02), "padding")
        return {
            "target": target,
            "obstacle": obstacle,
            "padding": padding,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target = _resolve_object_strict(context, validated["target"], "target")
        obstacle = _deepest_overlap(context, target, validated.get("obstacle"))
        target_bounds = world_bounds(target)
        obstacle_bounds = world_bounds(obstacle)
        overlap_extents = bounds_intersection_extents(target_bounds, obstacle_bounds)
        if not all(component > CONTACT_TOLERANCE for component in overlap_extents):
            raise ToolExecutionError(f"'{target.name}' does not deeply overlap '{obstacle.name}'")
        axis = min(range(3), key=lambda index: overlap_extents[index])
        target_center = bounds_center(target_bounds)
        obstacle_center = bounds_center(obstacle_bounds)
        direction = -1.0 if target_center[axis] < obstacle_center[axis] else 1.0
        location = [float(component) for component in getattr(target, "location", target_center)[:3]]
        location[axis] += direction * (float(overlap_extents[axis]) + float(validated["padding"]))
        resolved_location = (location[0], location[1], location[2])
        _set_object_location(target, resolved_location)
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=obstacle.name,
            location=resolved_location,
            placement_mode="overlap_resolution",
            placement_reason=f"Moved '{target.name}' away from '{obstacle.name}' along axis {axis}.",
        )


@register_tool
class FitInsideTool(BaseTool):
    name = "object.fit_inside"
    description = "Clamp an object's AABB footprint inside a named container or support bounds."
    input_schema = {
        "target": {"type": "string", "required": True},
        "container": {"type": "string", "required": True},
        "padding": {"type": "number", "required": False},
        "align_bottom": {"type": "boolean", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        container = _required_object_name(params.get("container"), "container")
        padding = _normalize_distance(params.get("padding", 0.0), "padding")
        align_bottom = params.get("align_bottom", False)
        if not isinstance(align_bottom, bool):
            raise ToolValidationError("'align_bottom' must be a boolean")
        return {
            "target": target,
            "container": container,
            "padding": padding,
            "align_bottom": align_bottom,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target, container = _resolve_target_and_reference(
            context,
            {"target": validated["target"], "reference": validated["container"]},
        )
        target_bounds = world_bounds(target)
        container_bounds = world_bounds(container)
        target_half_extents = bounds_half_extents(target_bounds)
        container_min = container_bounds["min"]
        container_max = container_bounds["max"]
        location = [float(component) for component in getattr(target, "location", bounds_center(target_bounds))[:3]]
        for axis in (0, 1):
            minimum = float(container_min[axis]) + float(target_half_extents[axis]) + float(validated["padding"])
            maximum = float(container_max[axis]) - float(target_half_extents[axis]) - float(validated["padding"])
            if minimum > maximum:
                location[axis] = (float(container_min[axis]) + float(container_max[axis])) / 2.0
            else:
                location[axis] = min(max(location[axis], minimum), maximum)
        if bool(validated["align_bottom"]) or is_floor_like(container):
            location[2] = float(container_max[2]) + float(target_half_extents[2])
        elif not bounds_contains_xy(container_bounds, target_bounds):
            z_min = float(container_min[2]) + float(target_half_extents[2])
            z_max = float(container_max[2]) - float(target_half_extents[2])
            if z_min <= z_max:
                location[2] = min(max(location[2], z_min), z_max)
        resolved_location = (location[0], location[1], location[2])
        _set_object_location(target, resolved_location)
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=container.name,
            location=resolved_location,
            placement_mode="fit_inside",
            placement_reason=f"Fit '{target.name}' inside '{container.name}'.",
        )


@register_tool
class PlaceRelativeTool(BaseTool):
    name = "object.place_relative"
    description = "Place one object at a stable AABB-based relation from a reference object."
    input_schema = {
        "target": {"type": "string", "required": True},
        "reference": {"type": "string", "required": True},
        "relation": {"type": "string", "required": True, "enum": sorted(_SUPPORTED_RELATIONS)},
        "distance": {"type": "number", "required": True},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        reference = _required_object_name(params.get("reference"), "reference")
        relation = _required_object_name(params.get("relation"), "relation")
        normalized_relation = str(relation).strip().lower()
        if normalized_relation not in _SUPPORTED_RELATIONS:
            raise ToolValidationError(f"'relation' must be one of {sorted(_SUPPORTED_RELATIONS)}")
        distance = _normalize_distance(params.get("distance"), "distance")
        return {
            "target": target,
            "reference": reference,
            "relation": normalized_relation,
            "distance": distance,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target, reference = _resolve_target_and_reference(context, validated)
        location = place_relative_location(
            world_bounds(target),
            world_bounds(reference),
            relation=validated["relation"],
            distance=float(validated["distance"]),
            current_location=getattr(target, "location", None),
        )
        target.location = location
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=reference.name,
            location=location,
            placement_mode="relative_contact",
            placement_reason=f"Placed '{target.name}' {validated['relation']} '{reference.name}'.",
        )


@register_tool
class AlignToTool(BaseTool):
    name = "object.align_to"
    description = "Align one object's AABB center to a reference object along a chosen axis."
    input_schema = {
        "target": {"type": "string", "required": True},
        "reference": {"type": "string", "required": True},
        "axis": {"type": "string", "required": True, "enum": sorted(_SUPPORTED_AXES)},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        reference = _required_object_name(params.get("reference"), "reference")
        axis = _required_object_name(params.get("axis"), "axis")
        normalized_axis = str(axis).strip().lower()
        if normalized_axis not in _SUPPORTED_AXES:
            raise ToolValidationError(f"'axis' must be one of {sorted(_SUPPORTED_AXES)}")
        return {
            "target": target,
            "reference": reference,
            "axis": normalized_axis,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target, reference = _resolve_target_and_reference(context, validated)
        location = align_to_location(
            world_bounds(target),
            world_bounds(reference),
            axis=validated["axis"],
        )
        target.location = location
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=reference.name,
            location=location,
            placement_mode="axis_alignment",
            placement_reason=f"Aligned '{target.name}' to '{reference.name}' on the {validated['axis'].upper()} axis.",
        )


@register_tool
class PlaceAtAnchorTool(BaseTool):
    name = "object.place_at_anchor"
    description = "Place one object at a deterministic spatial anchor derived from current scene geometry."
    input_schema = {
        "target": {"type": "string", "required": True},
        "anchor": {"type": "string", "required": True},
        "align_bottom": {"type": "boolean", "required": False},
        "offset": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        anchor = _required_object_name(params.get("anchor"), "anchor")
        align_bottom = params.get("align_bottom", True)
        if not isinstance(align_bottom, bool):
            raise ToolValidationError("'align_bottom' must be a boolean")
        offset = _ZERO_OFFSET
        if params.get("offset") is not None:
            offset = validate_vector3(params["offset"], "offset")
        return {
            "target": target,
            "anchor": anchor,
            "align_bottom": align_bottom,
            "offset": offset,
        }

    def execute(self, context: Any, params: dict[str, Any]) -> ToolExecutionResult:
        if bpy is None:
            raise ToolExecutionError("Blender Python API is unavailable")
        ensure_object_mode(context)
        validated = self.validate_params(params)
        target = _resolve_object_strict(context, validated["target"], "target")
        anchor = _resolve_anchor(context, validated["anchor"])
        location = _anchor_location_for_target(
            target,
            anchor,
            align_bottom=validated["align_bottom"],
            offset=validated["offset"],
        )
        target.location = location
        _update_view_layer(context)
        anchor_name = str(anchor.get("name", validated["anchor"]))
        return _placement_result(
            target=target,
            reference_name=anchor_name,
            anchor_name=anchor_name,
            location=location,
            placement_mode="anchor_placement",
            placement_reason=f"Placed '{target.name}' at spatial anchor '{anchor_name}'.",
        )
