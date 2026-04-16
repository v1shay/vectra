from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vectra.tools.spatial import (
    bounds_half_extents,
    find_floor_object,
    object_type,
    place_on_surface_location,
    primitive_bounds,
    world_bounds,
)

from .models import AssumptionRecord, DirectorContext

_VAGUE_REFERENCES = {
    "it",
    "them",
    "both",
    "all",
    "another",
    "here",
    "there",
    "center",
    "middle",
    "origin",
    "world origin",
}


def _scene_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    objects = scene_state.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict)]


def _object_by_name(scene_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        obj["name"]: obj
        for obj in _scene_objects(scene_state)
        if isinstance(obj.get("name"), str)
    }


def _mesh_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        obj
        for obj in _scene_objects(scene_state)
        if object_type(obj) == "MESH"
    ]


def _group_by_name(scene_state: dict[str, Any]) -> dict[str, list[str]]:
    groups = scene_state.get("groups", [])
    if not isinstance(groups, list):
        return {}
    resolved: dict[str, list[str]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = group.get("name")
        object_names = group.get("object_names")
        if isinstance(name, str) and isinstance(object_names, list):
            cleaned = [value for value in object_names if isinstance(value, str) and value.strip()]
            if cleaned:
                resolved[name] = cleaned
    return resolved


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _active_object_name(scene_state: dict[str, Any]) -> str | None:
    active = scene_state.get("active_object")
    if isinstance(active, str) and active.strip():
        return active
    return None


def _selected_object_names(scene_state: dict[str, Any]) -> list[str]:
    selected = scene_state.get("selected_objects")
    if not isinstance(selected, list):
        return []
    return [value for value in selected if isinstance(value, str) and value.strip()]


def _history_last_objects(history: list[dict[str, Any]]) -> list[str]:
    collected: list[str] = []
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        details = entry.get("details", {})
        if not isinstance(details, dict):
            continue
        outputs = details.get("results")
        if isinstance(outputs, list):
            for result in outputs:
                if not isinstance(result, dict):
                    continue
                result_outputs = result.get("outputs", {})
                if isinstance(result_outputs, dict):
                    object_name = result_outputs.get("object_name")
                    if isinstance(object_name, str) and object_name.strip():
                        collected.append(object_name)
        metadata = details.get("metadata", {})
        if isinstance(metadata, dict):
            affected = metadata.get("affected_object_names")
            if isinstance(affected, list):
                for object_name in affected:
                    if isinstance(object_name, str) and object_name.strip():
                        collected.append(object_name)
        if collected:
            break
    return collected


def _scene_centroid(scene_state: dict[str, Any]) -> list[float]:
    objects = _scene_objects(scene_state)
    locations = [
        _vector3(obj.get("location"))
        for obj in objects
    ]
    resolved = [vector for vector in locations if vector is not None]
    if not resolved:
        return [0.0, 0.0, 0.0]
    count = float(len(resolved))
    return [
        sum(vector[index] for vector in resolved) / count
        for index in range(3)
    ]


def _bounds_width(obj: dict[str, Any]) -> float:
    dimensions = _vector3(obj.get("dimensions"))
    if dimensions is None:
        return 2.0
    return max(float(dimensions[0]), 0.5)


@dataclass
class ResolutionResult:
    value: Any
    assumptions: list[AssumptionRecord]
    metadata: dict[str, Any]


class ReferenceResolver:
    def __init__(self, context: DirectorContext) -> None:
        self.context = context
        self._scene_map = _object_by_name(context.scene_state)
        self._group_map = _group_by_name(context.scene_state)
        self._mesh_objects = _mesh_objects(context.scene_state)
        floor_candidate = find_floor_object(self._mesh_objects)
        self._floor_name = str(floor_candidate.get("name", "")).strip() if isinstance(floor_candidate, dict) else None

    def resolve_target(self, raw_target: Any) -> ResolutionResult:
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {}
        if isinstance(raw_target, str) and raw_target.strip():
            stripped = raw_target.strip()
            exact = self._scene_map.get(stripped)
            if exact is not None:
                metadata["anchor"] = stripped
                return ResolutionResult(stripped, assumptions, metadata)
            if stripped in self._group_map:
                assumptions.append(
                    AssumptionRecord(
                        key="target",
                        value=self._group_map[stripped][0],
                        reason=f"Resolved group '{stripped}' to its first object for a single-target operation.",
                    )
                )
                metadata["anchor"] = stripped
                return ResolutionResult(self._group_map[stripped][0], assumptions, metadata)

            lowered = stripped.lower()
            if lowered in _VAGUE_REFERENCES:
                raw_target = None
            else:
                for name in self._scene_map:
                    if lowered == name.lower() or lowered in name.lower():
                        assumptions.append(
                            AssumptionRecord(
                                key="target",
                                value=name,
                                reason=f"Resolved vague reference '{stripped}' to the closest matching object.",
                            )
                        )
                        metadata["anchor"] = name
                        return ResolutionResult(name, assumptions, metadata)

        active_object = _active_object_name(self.context.scene_state)
        if active_object:
            assumptions.append(
                AssumptionRecord(
                    key="target",
                    value=active_object,
                    reason="Used the active object as the best available target.",
                )
            )
            metadata["anchor"] = active_object
            return ResolutionResult(active_object, assumptions, metadata)

        selected = _selected_object_names(self.context.scene_state)
        if selected:
            assumptions.append(
                AssumptionRecord(
                    key="target",
                    value=selected[0],
                    reason="Used the first selected object as the best available target.",
                )
            )
            metadata["anchor"] = selected[0]
            return ResolutionResult(selected[0], assumptions, metadata)

        last_objects = _history_last_objects(self.context.history)
        if last_objects:
            assumptions.append(
                AssumptionRecord(
                    key="target",
                    value=last_objects[0],
                    reason="Used the most recently affected object as the target.",
                )
            )
            metadata["anchor"] = last_objects[0]
            return ResolutionResult(last_objects[0], assumptions, metadata)

        return ResolutionResult(
            None,
            [
                AssumptionRecord(
                    key="target",
                    value=None,
                    reason="No matching object was available, so execution will fall back to the tool default.",
                )
            ],
            {"anchor": "none"},
        )

    def resolve_required_target(self, raw_target: Any, field_name: str) -> ResolutionResult:
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {}
        if isinstance(raw_target, str) and raw_target.strip():
            stripped = raw_target.strip()
            exact = self._scene_map.get(stripped)
            if exact is not None:
                metadata["anchor"] = stripped
                return ResolutionResult(stripped, assumptions, metadata)
            if stripped in self._group_map:
                assumptions.append(
                    AssumptionRecord(
                        key=field_name,
                        value=self._group_map[stripped][0],
                        reason=f"Resolved group '{stripped}' to its first object for a required spatial relation target.",
                    )
                )
                metadata["anchor"] = stripped
                return ResolutionResult(self._group_map[stripped][0], assumptions, metadata)

            lowered = stripped.lower()
            if lowered not in _VAGUE_REFERENCES:
                for name in self._scene_map:
                    if lowered == name.lower() or lowered in name.lower():
                        assumptions.append(
                            AssumptionRecord(
                                key=field_name,
                                value=name,
                                reason=f"Resolved required spatial relation reference '{stripped}' to the closest matching object.",
                            )
                        )
                        metadata["anchor"] = name
                        return ResolutionResult(name, assumptions, metadata)

        assumptions.append(
            AssumptionRecord(
                key=field_name,
                value=None,
                reason=f"Spatial relation field '{field_name}' requires an explicit resolvable object; no fallback target was used.",
            )
        )
        return ResolutionResult(None, assumptions, {"anchor": "unresolved_required"})

    def _floor_record(self) -> dict[str, Any] | None:
        if self._floor_name:
            return self._scene_map.get(self._floor_name)
        return None

    def _grounded_primitive_location(
        self,
        primitive_type: str,
        *,
        scale: Any = None,
    ) -> ResolutionResult:
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {}
        target_bounds = primitive_bounds(
            primitive_type,
            scale=_vector3(scale) if scale is not None else None,
        )
        target_half_extents = bounds_half_extents(target_bounds)
        floor_record = self._floor_record()

        if floor_record is not None:
            guessed = place_on_surface_location(
                target_bounds,
                world_bounds(floor_record),
                surface="top",
                offset=0.0,
            )
            assumptions.append(
                AssumptionRecord(
                    key="location",
                    value=[float(component) for component in guessed],
                    reason=f"Placed the new object on the floor '{self._floor_name}' instead of guessing a raw coordinate.",
                )
            )
            metadata["anchor"] = self._floor_name or "floor"
            return ResolutionResult(list(guessed), assumptions, metadata)

        if self._mesh_objects:
            centroid = _scene_centroid(self.context.scene_state)
            guessed = [centroid[0], centroid[1], target_half_extents[2]]
            assumptions.append(
                AssumptionRecord(
                    key="location",
                    value=guessed,
                    reason="No floor anchor was available, so used the scene centroid with grounded Z instead of scene-footprint or world-origin placement.",
                )
            )
            metadata["anchor"] = "scene_centroid_no_floor"
            return ResolutionResult(guessed, assumptions, metadata)

        guessed = [0.0, 0.0, target_half_extents[2]]
        assumptions.append(
            AssumptionRecord(
                key="location",
                value=guessed,
                reason="Used grounded bootstrap placement at z=0 for the first geometric object.",
            )
        )
        metadata["anchor"] = "bootstrap_ground"
        return ResolutionResult(guessed, assumptions, metadata)

    def resolve_location(
        self,
        tool_name: str,
        raw_location: Any,
        *,
        target_name: str | None = None,
        primitive_type: str | None = None,
        scale: Any = None,
    ) -> ResolutionResult:
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {}
        location = _vector3(raw_location)
        if location is not None:
            metadata["anchor"] = "explicit"
            return ResolutionResult(location, assumptions, metadata)

        if tool_name == "mesh.create_primitive":
            return self._grounded_primitive_location(primitive_type or "cube", scale=scale)

        if target_name and target_name in self._scene_map:
            target = self._scene_map[target_name]
            base = _vector3(target.get("location")) or [0.0, 0.0, 0.0]
            width = _bounds_width(target)
            if tool_name == "mesh.create_primitive":
                guessed = [base[0] + width + 0.5, base[1], base[2]]
            elif tool_name == "light.create":
                guessed = [base[0] + 4.0, base[1] - 4.0, base[2] + 6.0]
            elif tool_name == "camera.ensure":
                guessed = [base[0] + 8.0, base[1] - 8.0, base[2] + 6.0]
            else:
                guessed = list(base)
            assumptions.append(
                AssumptionRecord(
                    key="location",
                    value=guessed,
                    reason=f"Inferred a useful location from target '{target_name}'.",
                )
            )
            metadata["anchor"] = target_name
            return ResolutionResult(guessed, assumptions, metadata)

        centroid = _scene_centroid(self.context.scene_state)
        if tool_name == "mesh.create_primitive":
            guessed = [centroid[0], centroid[1], centroid[2]]
        elif tool_name == "light.create":
            guessed = [centroid[0] + 4.0, centroid[1] - 4.0, centroid[2] + 6.0]
        elif tool_name == "camera.ensure":
            guessed = [centroid[0] + 8.0, centroid[1] - 8.0, centroid[2] + 6.0]
        else:
            guessed = centroid
        assumptions.append(
            AssumptionRecord(
                key="location",
                value=guessed,
                reason="Used the scene centroid or world origin as a safe default location.",
            )
        )
        metadata["anchor"] = "scene_centroid" if any(component != 0.0 for component in centroid) else "world_origin"
        return ResolutionResult(guessed, assumptions, metadata)

    def resolve_objects(self, raw_objects: Any) -> ResolutionResult:
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {}
        resolved: list[str] = []
        if isinstance(raw_objects, list):
            for item in raw_objects:
                target = self.resolve_target(item)
                if isinstance(target.value, str):
                    resolved.append(target.value)
                    assumptions.extend(target.assumptions)
        elif isinstance(raw_objects, str) and raw_objects.strip() in self._group_map:
            group_name = raw_objects.strip()
            assumptions.append(
                AssumptionRecord(
                    key="objects",
                    value=self._group_map[group_name],
                    reason=f"Resolved group '{group_name}' to its member objects.",
                )
            )
            metadata["anchor"] = group_name
            return ResolutionResult(self._group_map[group_name], assumptions, metadata)
        if resolved:
            metadata["anchor"] = "explicit_or_resolved"
            return ResolutionResult(resolved, assumptions, metadata)

        selected = _selected_object_names(self.context.scene_state)
        if selected:
            assumptions.append(
                AssumptionRecord(
                    key="objects",
                    value=selected,
                    reason="Used the selected objects because no explicit object list was provided.",
                )
            )
            metadata["anchor"] = "selected_objects"
            return ResolutionResult(selected, assumptions, metadata)

        last_objects = _history_last_objects(self.context.history)
        if last_objects:
            assumptions.append(
                AssumptionRecord(
                    key="objects",
                    value=last_objects,
                    reason="Used the most recently affected objects as the group target.",
                )
            )
            metadata["anchor"] = "last_created"
            return ResolutionResult(last_objects, assumptions, metadata)

        if self._group_map:
            first_group_name = next(iter(self._group_map))
            assumptions.append(
                AssumptionRecord(
                    key="objects",
                    value=self._group_map[first_group_name],
                    reason=f"Used the current group '{first_group_name}' as the best available object set.",
                )
            )
            metadata["anchor"] = first_group_name
            return ResolutionResult(self._group_map[first_group_name], assumptions, metadata)

        return ResolutionResult([], assumptions, {"anchor": "none"})
