from __future__ import annotations

from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from .base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from .helpers import ensure_object_mode, normalize_optional_string, resolve_object
from .registry import register_tool
from .spatial import (
    align_to_location,
    place_against_location,
    place_on_surface_location,
    place_relative_location,
    world_bounds,
)

_SUPPORTED_SURFACES = {"top", "bottom", "left", "right", "front", "back"}
_SUPPORTED_RELATIONS = {"above", "below", "behind", "in_front_of", "left_of", "next_to", "right_of"}
_SUPPORTED_AXES = {"x", "y", "z"}


def _normalize_distance(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ToolValidationError(f"'{field_name}' must be numeric")
    return float(value)


def _resolved_bounds_payload(obj: Any) -> dict[str, list[float]]:
    bounds = world_bounds(obj)
    return {
        "min": [float(component) for component in bounds["min"]],
        "max": [float(component) for component in bounds["max"]],
    }


def _placement_result(
    *,
    target: Any,
    reference: Any,
    location: tuple[float, float, float],
    placement_mode: str,
    placement_reason: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        outputs={
            "object_name": target.name,
            "object_names": [target.name],
            "placement_mode": placement_mode,
            "placement_reason": placement_reason,
            "reference_object": reference.name,
            "resolved_location": [float(component) for component in location],
            "resolved_bounds": _resolved_bounds_payload(target),
        },
        message=f"Placed '{target.name}' using {placement_mode}",
    )


def _resolve_target_and_reference(context: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    target = resolve_object(context, params.get("target"))
    reference = resolve_object(context, params.get("reference"))
    if target is None:
        raise ToolExecutionError("No spatial placement target could be resolved")
    if reference is None:
        raise ToolExecutionError("No spatial placement reference could be resolved")
    if target == reference:
        raise ToolExecutionError("The target and reference objects must be different")
    return target, reference


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
        target = normalize_optional_string(params.get("target"), "target")
        reference = normalize_optional_string(params.get("reference"), "reference")
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
        return _placement_result(
            target=target,
            reference=reference,
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
        target = normalize_optional_string(params.get("target"), "target")
        reference = normalize_optional_string(params.get("reference"), "reference")
        side = normalize_optional_string(params.get("side"), "side")
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
        )
        target.location = location
        return _placement_result(
            target=target,
            reference=reference,
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
        target = normalize_optional_string(params.get("target"), "target")
        reference = normalize_optional_string(params.get("reference"), "reference")
        relation = normalize_optional_string(params.get("relation"), "relation")
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
        )
        target.location = location
        return _placement_result(
            target=target,
            reference=reference,
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
        target = normalize_optional_string(params.get("target"), "target")
        reference = normalize_optional_string(params.get("reference"), "reference")
        axis = normalize_optional_string(params.get("axis"), "axis")
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
        return _placement_result(
            target=target,
            reference=reference,
            location=location,
            placement_mode="axis_alignment",
            placement_reason=f"Aligned '{target.name}' to '{reference.name}' on the {validated['axis'].upper()} axis.",
        )
