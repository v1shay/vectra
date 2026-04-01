from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, Field


IntentStatus = Literal["ok", "no_action"]
IntentAction = Literal["create", "transform"]
IntentTransformKind = Literal["location", "rotation", "scale"]


_MAGNITUDE_DEFAULTS = {
    "a bit": 1.0,
    "bit": 1.0,
    "a little": 1.0,
    "little": 1.0,
    "slightly": 1.0,
    "a couple": 2.0,
    "couple": 2.0,
}
_DIRECTION_ALIASES = {
    "forward": "forward",
    "forwards": "forward",
    "back": "backward",
    "backward": "backward",
    "backwards": "backward",
    "up": "up",
    "upward": "up",
    "upwards": "up",
    "down": "down",
    "downward": "down",
    "downwards": "down",
    "into_ground": "down",
    "into the ground": "down",
    "ground": "down",
    "left": "left",
    "right": "right",
}
_PRONOUN_TARGETS = {"it", "this", "that", "this thing", "that thing", "this object", "that object"}
_TARGET_FILLER_WORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "thing",
    "object",
    "obj",
    "one",
    "stuff",
    "shit",
}
_AXIS_ALIASES = {
    "x": "x",
    "y": "y",
    "z": "z",
    "horizontal": "x",
    "vertical": "z",
}
_PROMPT_DIRECTION_PHRASES = {
    "into the ground": "down",
    "backward": "backward",
    "backwards": "backward",
    "back": "backward",
    "forward": "forward",
    "forwards": "forward",
    "upward": "up",
    "upwards": "up",
    "up": "up",
    "downward": "down",
    "downwards": "down",
    "down": "down",
    "left": "left",
    "right": "right",
}
_PROMPT_PRIMITIVE_PHRASES = {
    "uv sphere": "uv_sphere",
    "sphere": "uv_sphere",
    "ball": "uv_sphere",
    "plane": "plane",
    "cube": "cube",
    "box": "cube",
}


class IntentStep(BaseModel):
    action: IntentAction
    target: str | None = None
    target_name: str | None = None
    target_ref: str | None = None
    primitive_type: str | None = None
    direction: str | None = None
    magnitude: float | None = None
    magnitude_qualifier: str | None = None
    transform_kind: IntentTransformKind | None = None
    axis: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class IntentEnvelope(BaseModel):
    status: IntentStatus = "no_action"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    steps: list[IntentStep] = Field(default_factory=list)


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().lower().split())
    return normalized or None


def is_pronoun_target(value: str | None) -> bool:
    normalized = _normalize_text(value)
    return normalized in _PRONOUN_TARGETS


def _sanitize_target(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None

    tokens = re.findall(r"[a-z0-9_]+", normalized)
    filtered = [token for token in tokens if token not in _TARGET_FILLER_WORDS]
    if not filtered:
        return None
    return " ".join(filtered)


def _scene_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    objects = scene_state.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict)]


def _exact_name_match(scene_state: dict[str, Any], raw_target: str | None) -> str | None:
    normalized_target = _normalize_text(raw_target)
    if normalized_target is None:
        return None

    for obj in _scene_objects(scene_state):
        object_name = obj.get("name")
        if isinstance(object_name, str) and object_name.lower() == normalized_target:
            return object_name
    return None


def _fuzzy_name_match(scene_state: dict[str, Any], raw_target: str | None) -> str | None:
    sanitized_target = _sanitize_target(raw_target)
    if sanitized_target is None:
        return None

    for obj in _scene_objects(scene_state):
        object_name = obj.get("name")
        if not isinstance(object_name, str):
            continue
        lowered_name = object_name.lower()
        if sanitized_target == lowered_name or sanitized_target in lowered_name:
            return object_name
    return None


def _active_object_name(scene_state: dict[str, Any]) -> str | None:
    active_object = scene_state.get("active_object")
    return active_object if isinstance(active_object, str) and active_object.strip() else None


def _single_selected_object_name(scene_state: dict[str, Any]) -> str | None:
    selected_objects = scene_state.get("selected_objects")
    if not isinstance(selected_objects, list) or len(selected_objects) != 1:
        return None

    selected_name = selected_objects[0]
    if isinstance(selected_name, str) and selected_name.strip():
        return selected_name
    return None


def _resolve_scene_target(scene_state: dict[str, Any], step: IntentStep) -> str | None:
    exact_name = _exact_name_match(scene_state, step.target_name or step.target)
    if exact_name is not None:
        return exact_name

    target_ref = _normalize_text(step.target_ref)
    if target_ref in {"active_object", "active"}:
        return _active_object_name(scene_state)
    if target_ref in {"selected_object", "selected"}:
        return _single_selected_object_name(scene_state)

    raw_target = _normalize_text(step.target)
    if raw_target in _PRONOUN_TARGETS:
        return _active_object_name(scene_state) or _single_selected_object_name(scene_state)

    fuzzy_name = _fuzzy_name_match(scene_state, step.target_name or step.target)
    if fuzzy_name is not None:
        return fuzzy_name

    return None


def _normalize_direction(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None
    return _DIRECTION_ALIASES.get(normalized, normalized)


def _normalize_axis(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None
    return _AXIS_ALIASES.get(normalized, normalized)


def _normalize_magnitude(step: IntentStep) -> float | None:
    if step.magnitude is not None:
        return float(step.magnitude)

    qualifier = _normalize_text(step.magnitude_qualifier)
    if qualifier is None:
        return None
    return _MAGNITUDE_DEFAULTS.get(qualifier)


def _prompt_directions(prompt: str) -> set[str]:
    normalized_prompt = f" {_normalize_text(prompt) or ''} "
    matches: set[str] = set()
    for phrase, direction in _PROMPT_DIRECTION_PHRASES.items():
        if f" {phrase} " in normalized_prompt:
            matches.add(direction)
    return matches


def _prompt_primitives(prompt: str) -> set[str]:
    normalized_prompt = f" {_normalize_text(prompt) or ''} "
    matches: set[str] = set()
    for phrase, primitive in _PROMPT_PRIMITIVE_PHRASES.items():
        if f" {phrase} " in normalized_prompt:
            matches.add(primitive)
    return matches


def _normalize_create_step(step: IntentStep) -> IntentStep:
    primitive_type = _normalize_text(step.primitive_type)
    return step.model_copy(
        update={
            "primitive_type": primitive_type,
            "target": _normalize_text(step.target),
            "target_name": _normalize_text(step.target_name),
            "target_ref": _normalize_text(step.target_ref),
        }
    )


def _normalize_transform_step(
    step: IntentStep,
    *,
    scene_state: dict[str, Any],
) -> IntentStep | None:
    transform_kind = step.transform_kind or "location"
    direction = _normalize_direction(step.direction)
    magnitude = _normalize_magnitude(step)
    axis = _normalize_axis(step.axis)

    if transform_kind == "rotation":
        axis = axis or "z"

    target_name = _resolve_scene_target(scene_state, step)

    if target_name is None and _normalize_text(step.target_ref) not in {"previous_step", "previous_object"}:
        return None

    normalized = step.model_copy(
        update={
            "target": _normalize_text(step.target),
            "target_name": target_name,
            "target_ref": _normalize_text(step.target_ref),
            "direction": direction,
            "magnitude": magnitude,
            "transform_kind": transform_kind,
            "axis": axis,
        }
    )
    return normalized


def normalize_intent(
    intent: IntentEnvelope,
    *,
    prompt: str,
    scene_state: dict[str, Any],
    minimum_confidence: float,
) -> IntentEnvelope:
    if intent.status != "ok" or not intent.steps:
        return IntentEnvelope(status="no_action", confidence=float(intent.confidence), reason=intent.reason)

    prompt_directions = _prompt_directions(prompt)
    prompt_primitives = _prompt_primitives(prompt)
    normalized_steps: list[IntentStep] = []
    for step in intent.steps:
        step_confidence = float(intent.confidence if step.confidence is None else step.confidence)
        if step_confidence < minimum_confidence:
            return IntentEnvelope(
                status="no_action",
                confidence=float(intent.confidence),
                reason=f"Intent confidence below threshold for action '{step.action}'",
            )

        if step.action == "create":
            normalized_step = _normalize_create_step(step).model_copy(
                update={"confidence": step_confidence}
            )
            if normalized_step.primitive_type is None:
                return IntentEnvelope(
                    status="no_action",
                    confidence=float(intent.confidence),
                    reason="Create intent did not resolve to a supported primitive",
                )
            if normalized_step.primitive_type not in prompt_primitives:
                return IntentEnvelope(
                    status="no_action",
                    confidence=float(intent.confidence),
                    reason="Create intent was not grounded by an explicit primitive in the prompt",
                )
            normalized_steps.append(normalized_step)
            continue

        normalized_step = _normalize_transform_step(
            step.model_copy(update={"confidence": step_confidence}),
            scene_state=scene_state,
        )
        if normalized_step is None:
            return IntentEnvelope(
                status="no_action",
                confidence=float(intent.confidence),
                reason="Transform intent target could not be resolved safely",
            )

        unresolved_but_referenceable = (
            normalized_step.target_name is None
            and (
                normalized_step.target_ref in {"previous_step", "previous_object"}
                or is_pronoun_target(normalized_step.target)
            )
        )
        if normalized_step.target_name is None and not unresolved_but_referenceable:
            return IntentEnvelope(
                status="no_action",
                confidence=float(intent.confidence),
                reason="Transform intent target could not be resolved safely",
            )

        if normalized_step.transform_kind == "location":
            if normalized_step.direction is None:
                return IntentEnvelope(
                    status="no_action",
                    confidence=float(intent.confidence),
                    reason="Transform direction could not be resolved safely",
                )
            if normalized_step.direction not in prompt_directions:
                return IntentEnvelope(
                    status="no_action",
                    confidence=float(intent.confidence),
                    reason="Transform direction was not grounded by the prompt",
                )
            if normalized_step.magnitude is None:
                normalized_step = normalized_step.model_copy(update={"magnitude": 1.0})
        elif normalized_step.transform_kind == "rotation":
            if normalized_step.magnitude is None:
                return IntentEnvelope(
                    status="no_action",
                    confidence=float(intent.confidence),
                    reason="Rotation magnitude could not be resolved safely",
                )
        else:
            return IntentEnvelope(
                status="no_action",
                confidence=float(intent.confidence),
                reason=f"Unsupported transform kind '{normalized_step.transform_kind}'",
            )

        normalized_steps.append(normalized_step)

    return IntentEnvelope(
        status="ok",
        confidence=float(intent.confidence),
        reason=intent.reason,
        steps=normalized_steps,
    )


def build_direction_delta(direction: str, magnitude: float) -> tuple[float, float, float]:
    offsets = {
        "right": (magnitude, 0.0, 0.0),
        "left": (-magnitude, 0.0, 0.0),
        "forward": (0.0, magnitude, 0.0),
        "backward": (0.0, -magnitude, 0.0),
        "up": (0.0, 0.0, magnitude),
        "down": (0.0, 0.0, -magnitude),
    }
    try:
        return offsets[direction]
    except KeyError as exc:
        raise ValueError(f"Unsupported direction '{direction}'") from exc


def apply_vector_delta(base_vector: list[float], delta: tuple[float, float, float]) -> list[float]:
    return [
        float(base_vector[0]) + float(delta[0]),
        float(base_vector[1]) + float(delta[1]),
        float(base_vector[2]) + float(delta[2]),
    ]


def apply_rotation_delta(base_vector: list[float], *, axis: str, magnitude_degrees: float) -> list[float]:
    normalized_axis = _normalize_axis(axis)
    if normalized_axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported rotation axis '{axis}'")

    offset = math.radians(float(magnitude_degrees))
    current = [float(component) for component in base_vector]
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    current[axis_to_index[normalized_axis]] += offset
    return current


def sanitize_action_token(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "step"
