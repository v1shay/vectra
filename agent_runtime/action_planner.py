from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vectra.tools.registry import ToolRegistry, ToolRegistryError, get_default_registry

try:
    from .intent import (
        IntentEnvelope,
        IntentStep,
        apply_rotation_delta,
        apply_vector_delta,
        build_direction_delta,
        is_pronoun_target,
        sanitize_action_token,
    )
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from intent import (
        IntentEnvelope,
        IntentStep,
        apply_rotation_delta,
        apply_vector_delta,
        build_direction_delta,
        is_pronoun_target,
        sanitize_action_token,
    )


class ActionPlanningError(Exception):
    """Raised when normalized intent cannot be converted into a valid action plan."""


@dataclass(frozen=True)
class PlannedAction:
    action_id: str
    tool: str
    params: dict[str, Any]


_CREATE_TOOL_NAME = "mesh.create_primitive"
_TRANSFORM_TOOL_NAME = "object.transform"


def _get_registry() -> ToolRegistry:
    registry = get_default_registry()
    registry.discover()
    return registry


def _require_tool(registry: ToolRegistry, tool_name: str) -> None:
    try:
        registry.get(tool_name)
    except ToolRegistryError as exc:
        raise ActionPlanningError(f"Required tool '{tool_name}' is not registered") from exc


def _scene_object_by_name(scene_state: dict[str, Any], object_name: str) -> dict[str, Any]:
    objects = scene_state.get("objects", [])
    if not isinstance(objects, list):
        raise ActionPlanningError(f"Scene object '{object_name}' was not found")

    for obj in objects:
        if isinstance(obj, dict) and obj.get("name") == object_name:
            return obj
    raise ActionPlanningError(f"Scene object '{object_name}' was not found")


def _create_action_id(step_index: int, step: IntentStep, *, default_target: str) -> str:
    action_name = step.action
    target_token = sanitize_action_token(default_target)
    return f"step_{step_index + 1}_{action_name}_{target_token}"


def _plan_create_step(step_index: int, step: IntentStep) -> PlannedAction:
    primitive_type = step.primitive_type
    if primitive_type is None:
        raise ActionPlanningError("Create intent is missing primitive_type")

    action_id = _create_action_id(step_index, step, default_target=primitive_type)
    return PlannedAction(
        action_id=action_id,
        tool=_CREATE_TOOL_NAME,
        params={
            "primitive_type": primitive_type,
            "location": [0.0, 0.0, 0.0],
        },
    )


def _plan_transform_step(
    step_index: int,
    step: IntentStep,
    *,
    scene_state: dict[str, Any],
    last_create_action_id: str | None,
    last_create_location: list[float] | None,
) -> PlannedAction:
    transform_kind = step.transform_kind or "location"

    object_name_param: str | dict[str, str]
    target_token = "object"
    should_use_previous_step = step.target_ref in {"previous_step", "previous_object"} or (
        step.target_name is None and last_create_action_id is not None and is_pronoun_target(step.target)
    )
    if should_use_previous_step:
        if last_create_action_id is None:
            raise ActionPlanningError("Transform intent references a previous object, but none exists")
        object_name_param = {"$ref": f"{last_create_action_id}.object_name"}
        target_token = "previous"
        scene_object = None
    else:
        if step.target_name is None:
            raise ActionPlanningError("Transform intent is missing target_name")
        object_name_param = step.target_name
        target_token = step.target_name
        scene_object = _scene_object_by_name(scene_state, step.target_name)

    params: dict[str, Any] = {"object_name": object_name_param}
    if transform_kind == "location":
        if step.direction is None or step.magnitude is None:
            raise ActionPlanningError("Location transform requires direction and magnitude")
        if scene_object is None:
            if last_create_location is None:
                raise ActionPlanningError("Location transform requires a resolved scene object")
            base_vector = last_create_location
        else:
            base_vector = scene_object.get("location")
            if not isinstance(base_vector, list) or len(base_vector) != 3:
                raise ActionPlanningError("Scene object location is unavailable")
        delta = build_direction_delta(step.direction, float(step.magnitude))
        params["location"] = apply_vector_delta(base_vector, delta)
    elif transform_kind == "rotation":
        if step.magnitude is None:
            raise ActionPlanningError("Rotation transform requires magnitude")
        if scene_object is None:
            raise ActionPlanningError("Rotation transform requires a resolved scene object")
        base_vector = scene_object.get("rotation_euler")
        if not isinstance(base_vector, list) or len(base_vector) != 3:
            raise ActionPlanningError("Scene object rotation is unavailable")
        params["rotation_euler"] = apply_rotation_delta(
            base_vector,
            axis=step.axis or "z",
            magnitude_degrees=float(step.magnitude),
        )
    else:
        raise ActionPlanningError(f"Unsupported transform kind '{transform_kind}'")

    action_id = _create_action_id(step_index, step, default_target=target_token)
    return PlannedAction(action_id=action_id, tool=_TRANSFORM_TOOL_NAME, params=params)


def plan_actions(intent: IntentEnvelope, *, scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    if intent.status != "ok" or not intent.steps:
        return []

    registry = _get_registry()
    _require_tool(registry, _CREATE_TOOL_NAME)
    _require_tool(registry, _TRANSFORM_TOOL_NAME)

    planned_actions: list[PlannedAction] = []
    last_create_action_id: str | None = None
    last_create_location: list[float] | None = None
    for index, step in enumerate(intent.steps):
        if step.action == "create":
            planned = _plan_create_step(index, step)
            last_create_action_id = planned.action_id
            raw_location = planned.params.get("location", [0.0, 0.0, 0.0])
            last_create_location = [float(component) for component in raw_location]
        elif step.action == "transform":
            planned = _plan_transform_step(
                index,
                step,
                scene_state=scene_state,
                last_create_action_id=last_create_action_id,
                last_create_location=last_create_location,
            )
        else:  # pragma: no cover - IntentStep typing guards this
            raise ActionPlanningError(f"Unsupported intent action '{step.action}'")

        planned_actions.append(planned)

    return [
        {
            "action_id": action.action_id,
            "tool": action.tool,
            "params": action.params,
        }
        for action in planned_actions
    ]
