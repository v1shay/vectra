from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

try:
    from .scene_intent import ConstructionStep, SceneEntity, SceneGroup, SceneIntent, SceneRelationship
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from agent_runtime.scene_intent import ConstructionStep, SceneEntity, SceneGroup, SceneIntent, SceneRelationship

_DEFAULT_DIMENSIONS = {
    "cube": [2.0, 2.0, 2.0],
    "plane": [2.0, 2.0, 0.0],
    "uv_sphere": [2.0, 2.0, 2.0],
}
_DISPLAY_NAMES = {
    "cube": "Cube",
    "plane": "Plane",
    "uv_sphere": "Sphere",
}
_REFERENCE_PRONOUNS = {"them", "both", "all", "each"}


class ConstructionError(Exception):
    """Raised when a scene intent cannot be compiled safely."""


class RuntimeObjectState(BaseModel):
    logical_id: str | None = None
    runtime_name: str
    kind: str
    location: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_euler: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    dimensions: list[float] = Field(default_factory=lambda: [2.0, 2.0, 2.0])
    group_id: str | None = None


class ConstructionState(BaseModel):
    objects_by_logical_id: dict[str, list[RuntimeObjectState]] = Field(default_factory=dict)
    objects_by_runtime_name: dict[str, RuntimeObjectState] = Field(default_factory=dict)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    last_created: list[str] = Field(default_factory=list)
    last_group: str | None = None
    counters_by_kind: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class StepCompileResult:
    actions: list[dict[str, Any]] = field(default_factory=list)
    affected_logical_ids: list[str] = field(default_factory=list)
    affected_group_ids: list[str] = field(default_factory=list)
    consumed_budget: bool = False


@dataclass(frozen=True)
class CompiledConstructionPlan:
    actions: list[dict[str, Any]]
    state: ConstructionState
    steps: list[ConstructionStep]
    affected_logical_ids: list[str]
    affected_group_ids: list[str]
    continue_loop: bool
    expected_outcome: str


def _copy_object(raw_object: dict[str, Any]) -> RuntimeObjectState:
    name = str(raw_object.get("name", "Object")).strip() or "Object"
    kind = _infer_kind(raw_object)
    return RuntimeObjectState(
        runtime_name=name,
        kind=kind,
        location=_coerce_scene_vector(raw_object.get("location"), default=[0.0, 0.0, 0.0]),
        rotation_euler=_coerce_scene_vector(raw_object.get("rotation_euler"), default=[0.0, 0.0, 0.0]),
        scale=_coerce_scene_vector(raw_object.get("scale"), default=[1.0, 1.0, 1.0]),
        dimensions=_coerce_scene_vector(raw_object.get("dimensions"), default=_DEFAULT_DIMENSIONS[kind]),
    )


def _coerce_scene_vector(value: Any, *, default: list[float]) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        return [float(component) for component in default]
    return [float(component) for component in value]


def _infer_kind(raw_object: dict[str, Any]) -> str:
    lowered_name = str(raw_object.get("name", "")).strip().lower()
    if "plane" in lowered_name:
        return "plane"
    if "sphere" in lowered_name:
        return "uv_sphere"
    if "cube" in lowered_name or "box" in lowered_name or "stair" in lowered_name:
        return "cube"

    dimensions = raw_object.get("dimensions")
    if isinstance(dimensions, list) and len(dimensions) == 3:
        width, depth, height = (float(component) for component in dimensions)
        if abs(width - depth) < 0.001 and abs(depth - height) < 0.001:
            return "cube"
        if height < 0.25:
            return "plane"
    return "cube"


def _runtime_prefix(kind: str) -> str:
    return _DISPLAY_NAMES.get(kind, kind.title())


def _default_dimensions(kind: str) -> list[float]:
    return [float(component) for component in _DEFAULT_DIMENSIONS.get(kind, [2.0, 2.0, 2.0])]


def _vector_add(base: list[float], offset: list[float]) -> list[float]:
    return [float(base[index]) + float(offset[index]) for index in range(3)]


def _set_logical_binding(state: ConstructionState, logical_id: str, runtime_object: RuntimeObjectState) -> None:
    runtime_object.logical_id = logical_id
    state.objects_by_runtime_name[runtime_object.runtime_name] = runtime_object
    bound = state.objects_by_logical_id.setdefault(logical_id, [])
    if all(existing.runtime_name != runtime_object.runtime_name for existing in bound):
        bound.append(runtime_object)


def _seed_counters(state: ConstructionState) -> None:
    counters: dict[str, int] = {}
    for runtime_name, runtime_object in state.objects_by_runtime_name.items():
        prefix = _runtime_prefix(runtime_object.kind)
        if runtime_name.startswith(f"{prefix}_"):
            suffix = runtime_name.split("_", 1)[1]
            if suffix.isdigit():
                counters[runtime_object.kind] = max(counters.get(runtime_object.kind, 0), int(suffix))
    state.counters_by_kind = counters


def build_construction_state(scene_state: dict[str, Any]) -> ConstructionState:
    state = ConstructionState()
    objects = scene_state.get("objects", [])
    if isinstance(objects, list):
        for raw_object in objects:
            if not isinstance(raw_object, dict):
                continue
            runtime_object = _copy_object(raw_object)
            state.objects_by_runtime_name[runtime_object.runtime_name] = runtime_object
    _seed_counters(state)
    return state


def _entity_group_id(entity: SceneEntity) -> str | None:
    if entity.group_id:
        return entity.group_id
    if entity.quantity > 1:
        return f"group_{entity.logical_id}"
    return None


def _expanded_entity_ids(entity: SceneEntity) -> list[str]:
    if entity.quantity <= 1:
        return [entity.logical_id]
    return [f"{entity.logical_id}_{index}" for index in range(1, entity.quantity + 1)]


def _expanded_groups(intent: SceneIntent) -> list[SceneGroup]:
    groups: dict[str, SceneGroup] = {group.logical_id: group.model_copy(deep=True) for group in intent.groups}
    for entity in intent.entities:
        group_id = _entity_group_id(entity)
        if group_id is None:
            continue
        entity_ids = _expanded_entity_ids(entity)
        existing = groups.get(group_id)
        pronouns = ["them", "both", "all", "each"]
        if existing is None:
            groups[group_id] = SceneGroup(logical_id=group_id, entity_ids=entity_ids, pronouns=pronouns)
            continue
        merged_ids = list(existing.entity_ids)
        for entity_id in entity_ids:
            if entity_id not in merged_ids:
                merged_ids.append(entity_id)
        merged_pronouns = list(existing.pronouns)
        for pronoun in pronouns:
            if pronoun not in merged_pronouns:
                merged_pronouns.append(pronoun)
        groups[group_id] = existing.model_copy(update={"entity_ids": merged_ids, "pronouns": merged_pronouns})
    return list(groups.values())


def decompose_scene_intent(intent: SceneIntent) -> list[ConstructionStep]:
    if intent.construction_steps:
        return [step.model_copy(deep=True) for step in intent.construction_steps]

    steps: list[ConstructionStep] = []
    expanded_groups = _expanded_groups(intent)
    group_ids = {group.logical_id for group in expanded_groups}
    relationships = list(intent.relationships)

    if intent.metadata.get("pattern") == "staircase":
        staircase_offset = intent.metadata.get("stair_offset", [1.25, 0.0, 1.0])
        if isinstance(staircase_offset, list) and len(staircase_offset) == 3:
            for entity in intent.entities:
                entity_ids = _expanded_entity_ids(entity)
                for index in range(1, len(entity_ids)):
                    relationships.append(
                        SceneRelationship(
                            logical_id=f"{entity.logical_id}_stair_{index}",
                            relation_type="relative_offset",
                            source_id=entity_ids[index],
                            target_id=entity_ids[index - 1],
                            offset=[float(component) for component in staircase_offset],
                        )
                    )

    for entity in intent.entities:
        steps.append(
            ConstructionStep(
                logical_id=f"ensure_{entity.logical_id}",
                kind="ensure_entity",
                entity_id=entity.logical_id,
                group_id=_entity_group_id(entity),
            )
        )
        if entity.initial_transform and entity.initial_transform.offset is not None:
            steps.append(
                ConstructionStep(
                    logical_id=f"transform_{entity.logical_id}",
                    kind="apply_transform",
                    entity_id=entity.logical_id,
                    offset=[float(component) for component in entity.initial_transform.offset],
                    group_id=_entity_group_id(entity),
                )
            )
        group_id = _entity_group_id(entity)
        if group_id is not None and group_id in group_ids:
            steps.append(
                ConstructionStep(
                    logical_id=f"resolve_{group_id}",
                    kind="resolve_group",
                    entity_id=entity.logical_id,
                    group_id=group_id,
                )
            )

    for relationship in relationships:
        steps.append(
            ConstructionStep(
                logical_id=f"satisfy_{relationship.logical_id}",
                kind="satisfy_relation",
                entity_id=relationship.source_id,
                relationship_id=relationship.logical_id,
                group_id=relationship.target_group_id,
            )
        )
    return steps


def _entity_by_id(intent: SceneIntent, entity_id: str) -> SceneEntity:
    for entity in intent.entities:
        if entity.logical_id == entity_id:
            return entity
    raise ConstructionError(f"Unknown entity '{entity_id}'")


def _groups_by_id(intent: SceneIntent) -> dict[str, SceneGroup]:
    return {group.logical_id: group for group in _expanded_groups(intent)}


def _bind_existing_candidates(entity: SceneEntity, state: ConstructionState) -> None:
    pending_ids = [logical_id for logical_id in _expanded_entity_ids(entity) if logical_id not in state.objects_by_logical_id]
    if not pending_ids:
        return

    exact_reference_name = entity.reference_name or entity.display_name
    candidates: list[RuntimeObjectState] = []
    for runtime_object in state.objects_by_runtime_name.values():
        if runtime_object.logical_id is not None or runtime_object.kind != entity.kind:
            continue
        if exact_reference_name and runtime_object.runtime_name != exact_reference_name:
            continue
        candidates.append(runtime_object)

    candidates.sort(key=lambda item: item.runtime_name)
    for logical_id, runtime_object in zip(pending_ids, candidates):
        _set_logical_binding(state, logical_id, runtime_object)


def _next_runtime_name(kind: str, state: ConstructionState) -> str:
    next_index = int(state.counters_by_kind.get(kind, 0)) + 1
    state.counters_by_kind[kind] = next_index
    return f"{_runtime_prefix(kind)}_{next_index}"


def _create_action_for_entity(entity: SceneEntity, runtime_name: str, location: list[float], *, action_id: str) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "tool": "mesh.create_primitive",
        "params": {
            "primitive_type": entity.kind,
            "name": runtime_name,
            "location": [float(component) for component in location],
        },
    }


def _transform_action(runtime_name: str, location: list[float], *, action_id: str) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "tool": "object.transform",
        "params": {
            "object_name": runtime_name,
            "location": [float(component) for component in location],
        },
    }


def _resolve_group_members(group_id: str, state: ConstructionState) -> list[RuntimeObjectState]:
    logical_ids = state.groups.get(group_id, [])
    members: list[RuntimeObjectState] = []
    for logical_id in logical_ids:
        members.extend(state.objects_by_logical_id.get(logical_id, []))
    return sorted(members, key=lambda item: item.runtime_name)


def _resolve_entity_objects(entity_id: str, state: ConstructionState) -> list[RuntimeObjectState]:
    if entity_id in state.objects_by_logical_id:
        return sorted(state.objects_by_logical_id[entity_id], key=lambda item: item.runtime_name)

    prefix = f"{entity_id}_"
    expanded: list[RuntimeObjectState] = []
    for logical_id, objects in state.objects_by_logical_id.items():
        if logical_id.startswith(prefix):
            expanded.extend(objects)
    return sorted(expanded, key=lambda item: item.runtime_name)


def _register_group(entity: SceneEntity, state: ConstructionState) -> list[str]:
    group_id = _entity_group_id(entity)
    if group_id is None:
        return []
    logical_ids = _expanded_entity_ids(entity)
    state.groups[group_id] = logical_ids
    state.last_group = group_id
    return [group_id]


def _compile_ensure_entity(
    step: ConstructionStep,
    *,
    intent: SceneIntent,
    state: ConstructionState,
    max_creations: int | None,
) -> StepCompileResult:
    if step.entity_id is None:
        raise ConstructionError("ensure_entity step is missing entity_id")

    entity = _entity_by_id(intent, step.entity_id)
    _bind_existing_candidates(entity, state)
    pending_ids = [logical_id for logical_id in _expanded_entity_ids(entity) if logical_id not in state.objects_by_logical_id]
    if not pending_ids:
        affected_groups = _register_group(entity, state)
        return StepCompileResult(
            affected_logical_ids=_expanded_entity_ids(entity),
            affected_group_ids=affected_groups,
            consumed_budget=False,
        )

    if entity.source == "reference":
        raise ConstructionError(f"Entity '{entity.logical_id}' could not be resolved from the scene")

    creation_budget = len(pending_ids) if max_creations is None else max(1, min(max_creations, len(pending_ids)))
    actions: list[dict[str, Any]] = []
    affected_ids: list[str] = []
    for logical_id in pending_ids[:creation_budget]:
        runtime_name = _next_runtime_name(entity.kind, state)
        location = (
            [float(component) for component in entity.initial_transform.offset]
            if entity.initial_transform and entity.initial_transform.offset is not None
            else [0.0, 0.0, 0.0]
        )
        action_id = f"create_{logical_id}"
        actions.append(_create_action_for_entity(entity, runtime_name, location, action_id=action_id))
        runtime_object = RuntimeObjectState(
            logical_id=logical_id,
            runtime_name=runtime_name,
            kind=entity.kind,
            location=location,
            dimensions=_default_dimensions(entity.kind),
            group_id=_entity_group_id(entity),
        )
        _set_logical_binding(state, logical_id, runtime_object)
        affected_ids.append(logical_id)

    state.last_created = affected_ids
    affected_groups = _register_group(entity, state)
    return StepCompileResult(
        actions=actions,
        affected_logical_ids=affected_ids,
        affected_group_ids=affected_groups,
        consumed_budget=bool(actions),
    )


def _compile_apply_transform(
    step: ConstructionStep,
    *,
    intent: SceneIntent,
    state: ConstructionState,
) -> StepCompileResult:
    if step.entity_id is None:
        raise ConstructionError("apply_transform step is missing entity_id")
    if step.offset is None:
        raise ConstructionError("apply_transform step is missing offset")

    entity_objects = _resolve_entity_objects(step.entity_id, state)
    if not entity_objects:
        raise ConstructionError(f"Entity '{step.entity_id}' is not available for transform")

    actions: list[dict[str, Any]] = []
    affected_ids: list[str] = []
    for index, runtime_object in enumerate(entity_objects, start=1):
        target_location = [float(component) for component in step.offset]
        if all(abs(runtime_object.location[axis] - target_location[axis]) < 0.001 for axis in range(3)):
            continue
        runtime_object.location = [float(component) for component in target_location]
        actions.append(
            _transform_action(
                runtime_object.runtime_name,
                target_location,
                action_id=f"transform_{runtime_object.logical_id or step.entity_id}_{index}",
            )
        )
        if runtime_object.logical_id:
            affected_ids.append(runtime_object.logical_id)

    return StepCompileResult(
        actions=actions,
        affected_logical_ids=affected_ids,
        affected_group_ids=[step.group_id] if step.group_id else [],
        consumed_budget=bool(actions),
    )


def _resolve_relation_source_objects(relationship: SceneRelationship, state: ConstructionState) -> list[RuntimeObjectState]:
    source_objects = _resolve_entity_objects(relationship.source_id, state)
    if source_objects:
        return source_objects

    if relationship.source_id in state.groups:
        return _resolve_group_members(relationship.source_id, state)

    if relationship.source_id in _REFERENCE_PRONOUNS and state.last_group:
        return _resolve_group_members(state.last_group, state)
    return []


def _resolve_relation_target_objects(relationship: SceneRelationship, state: ConstructionState) -> list[RuntimeObjectState]:
    if relationship.target_id:
        targets = _resolve_entity_objects(relationship.target_id, state)
        if targets:
            return targets

    if relationship.target_group_id:
        targets = _resolve_group_members(relationship.target_group_id, state)
        if targets:
            return targets

    if relationship.target_id in _REFERENCE_PRONOUNS and state.last_group:
        return _resolve_group_members(state.last_group, state)
    return []


def _pairwise_relative_offsets(
    source_objects: list[RuntimeObjectState],
    target_objects: list[RuntimeObjectState],
    offset: list[float],
) -> list[tuple[RuntimeObjectState, list[float]]]:
    if target_objects:
        anchor = target_objects[0]
        return [(source_objects[0], _vector_add(anchor.location, offset))]

    results: list[tuple[RuntimeObjectState, list[float]]] = []
    for index in range(1, len(source_objects)):
        anchor = source_objects[index - 1]
        results.append((source_objects[index], _vector_add(anchor.location, offset)))
    return results


def _axis_offset_for_relation(
    relationship: SceneRelationship,
    source_object: RuntimeObjectState,
    target_object: RuntimeObjectState,
) -> list[float]:
    distance = float(relationship.distance or 0.5)
    source_dimensions = source_object.dimensions
    target_dimensions = target_object.dimensions

    if relationship.relation_type == "left_of":
        delta = -((source_dimensions[0] + target_dimensions[0]) / 2.0 + distance)
        return [target_object.location[0] + delta, target_object.location[1], target_object.location[2]]
    if relationship.relation_type == "right_of":
        delta = (source_dimensions[0] + target_dimensions[0]) / 2.0 + distance
        return [target_object.location[0] + delta, target_object.location[1], target_object.location[2]]
    if relationship.relation_type == "below":
        delta = -((source_dimensions[2] + target_dimensions[2]) / 2.0 + distance)
        return [target_object.location[0], target_object.location[1], target_object.location[2] + delta]

    if relationship.relation_type == "next_to":
        delta = (source_dimensions[0] + target_dimensions[0]) / 2.0 + distance
        return [target_object.location[0] + delta, target_object.location[1], target_object.location[2]]

    delta = (source_dimensions[2] + target_dimensions[2]) / 2.0 + distance
    if relationship.metadata.get("touching", True):
        delta = (source_dimensions[2] + target_dimensions[2]) / 2.0
    return [target_object.location[0], target_object.location[1], target_object.location[2] + delta]


def _compile_satisfy_relation(
    step: ConstructionStep,
    *,
    intent: SceneIntent,
    state: ConstructionState,
) -> StepCompileResult:
    if step.relationship_id is None:
        raise ConstructionError("satisfy_relation step is missing relationship_id")
    relationship = _relationship_lookup(intent, step.relationship_id)
    source_objects = _resolve_relation_source_objects(relationship, state)
    target_objects = _resolve_relation_target_objects(relationship, state)
    if not source_objects:
        raise ConstructionError(f"Relationship source '{relationship.source_id}' is unavailable")

    placements: list[tuple[RuntimeObjectState, list[float]]] = []
    if relationship.relation_type == "relative_offset":
        offset = relationship.offset or [3.0, 0.0, 0.0]
        placements = _pairwise_relative_offsets(source_objects, target_objects, offset)
    else:
        if not target_objects:
            raise ConstructionError(f"Relationship target for '{relationship.logical_id}' is unavailable")
        placements = [
            (source_objects[0], _axis_offset_for_relation(relationship, source_objects[0], target_objects[0]))
        ]

    actions: list[dict[str, Any]] = []
    affected_ids: list[str] = []
    for index, (runtime_object, target_location) in enumerate(placements, start=1):
        if all(abs(runtime_object.location[axis] - target_location[axis]) < 0.001 for axis in range(3)):
            continue
        runtime_object.location = [float(component) for component in target_location]
        actions.append(
            _transform_action(
                runtime_object.runtime_name,
                target_location,
                action_id=f"relation_{relationship.logical_id}_{index}",
            )
        )
        if runtime_object.logical_id:
            affected_ids.append(runtime_object.logical_id)

    return StepCompileResult(
        actions=actions,
        affected_logical_ids=affected_ids,
        affected_group_ids=[group_id for group_id, members in state.groups.items() if any(member in affected_ids for member in members)],
        consumed_budget=bool(actions),
    )


def _relationship_lookup(intent: SceneIntent, relationship_id: str) -> SceneRelationship:
    for relationship in intent.relationships:
        if relationship.logical_id == relationship_id:
            return relationship

    if intent.metadata.get("pattern") == "staircase":
        staircase_offset = intent.metadata.get("stair_offset", [1.25, 0.0, 1.0])
        for entity in intent.entities:
            entity_ids = _expanded_entity_ids(entity)
            for index in range(1, len(entity_ids)):
                generated = SceneRelationship(
                    logical_id=f"{entity.logical_id}_stair_{index}",
                    relation_type="relative_offset",
                    source_id=entity_ids[index],
                    target_id=entity_ids[index - 1],
                    offset=[float(component) for component in staircase_offset],
                )
                if generated.logical_id == relationship_id:
                    return generated

    raise ConstructionError(f"Unknown relationship '{relationship_id}'")


def _compile_resolve_group(
    step: ConstructionStep,
    *,
    intent: SceneIntent,
    state: ConstructionState,
) -> StepCompileResult:
    if step.group_id is None or step.entity_id is None:
        raise ConstructionError("resolve_group step is missing group_id or entity_id")
    entity = _entity_by_id(intent, step.entity_id)
    group_id = _entity_group_id(entity) or step.group_id
    state.groups[group_id] = _expanded_entity_ids(entity)
    state.last_group = group_id
    return StepCompileResult(
        affected_logical_ids=_expanded_entity_ids(entity),
        affected_group_ids=[group_id],
        consumed_budget=False,
    )


def _compile_step(
    step: ConstructionStep,
    *,
    intent: SceneIntent,
    state: ConstructionState,
    max_creations: int | None,
) -> StepCompileResult:
    if step.kind == "ensure_entity":
        return _compile_ensure_entity(step, intent=intent, state=state, max_creations=max_creations)
    if step.kind == "apply_transform":
        return _compile_apply_transform(step, intent=intent, state=state)
    if step.kind == "resolve_group":
        return _compile_resolve_group(step, intent=intent, state=state)
    if step.kind == "satisfy_relation":
        return _compile_satisfy_relation(step, intent=intent, state=state)
    raise ConstructionError(f"Unsupported construction step kind '{step.kind}'")


def _remaining_work(steps: list[ConstructionStep], *, intent: SceneIntent, state: ConstructionState) -> bool:
    probe_state = state.model_copy(deep=True)
    for step in steps:
        result = _compile_step(step, intent=intent, state=probe_state, max_creations=None)
        if result.actions:
            return True
    return False


def compile_construction_plan(
    intent: SceneIntent,
    *,
    scene_state: dict[str, Any],
    max_construction_steps: int | None,
) -> CompiledConstructionPlan:
    state = build_construction_state(scene_state)
    actions: list[dict[str, Any]] = []
    affected_logical_ids: list[str] = []
    affected_group_ids: list[str] = []
    steps = decompose_scene_intent(intent)
    consumed_budget = 0

    groups_by_id = _groups_by_id(intent)
    for group_id, group in groups_by_id.items():
        state.groups[group_id] = list(group.entity_ids)

    for step in steps:
        if max_construction_steps is not None and consumed_budget >= max_construction_steps:
            break

        creation_budget = None
        if max_construction_steps is not None:
            creation_budget = 1

        result = _compile_step(step, intent=intent, state=state, max_creations=creation_budget)
        if result.actions:
            actions.extend(result.actions)
        for logical_id in result.affected_logical_ids:
            if logical_id not in affected_logical_ids:
                affected_logical_ids.append(logical_id)
        for group_id in result.affected_group_ids:
            if group_id not in affected_group_ids:
                affected_group_ids.append(group_id)
        if result.consumed_budget:
            consumed_budget += 1

    continue_loop = _remaining_work(steps, intent=intent, state=state)
    expected_outcome = (
        "Scene intent is fully satisfied."
        if not continue_loop
        else "The scene is partially constructed and has remaining construction steps."
    )
    return CompiledConstructionPlan(
        actions=actions,
        state=state,
        steps=steps,
        affected_logical_ids=affected_logical_ids,
        affected_group_ids=affected_group_ids,
        continue_loop=continue_loop,
        expected_outcome=expected_outcome,
    )
