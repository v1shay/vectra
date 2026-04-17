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
    align_to_location,
    bounds_half_extents,
    bounds_to_lists,
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
    }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = super().validate_params(params)
        target = _required_object_name(params.get("target"), "target")
        reference = _required_object_name(params.get("reference"), "reference")
        surface = str(params.get("surface", "top")).strip().lower() or "top"
        if surface not in _SUPPORTED_SURFACES:
            raise ToolValidationError(f"'surface' must be one of {sorted(_SUPPORTED_SURFACES)}")
        offset = _normalize_distance(params.get("offset", 0.0), "offset")
        return {
            "target": target,
            "reference": reference,
            "surface": surface,
            "offset": offset,
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
        target.location = location
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
        return {
            "target": target,
            "reference": reference,
            "side": normalized_side,
            "offset": offset,
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
        target.location = location
        _update_view_layer(context)
        return _placement_result(
            target=target,
            reference_name=reference.name,
            location=location,
            placement_mode="against_contact",
            placement_reason=f"Placed '{target.name}' against the {validated['side']} side of '{reference.name}'.",
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
