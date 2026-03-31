from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vectra.tools.registry import ToolRegistry, ToolRegistryError, get_default_registry
from vectra.utils.logging import get_vectra_logger, log_structured

try:
    from .llm_client import LLMClientError, generate_actions
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from llm_client import LLMClientError, generate_actions


class PlannerValidationError(Exception):
    """Raised when planner actions fail structural validation."""


@dataclass(frozen=True)
class PlannerResult:
    status: str
    actions: list[dict[str, Any]]
    message: str


_PLANNER_LOGGER = get_vectra_logger("vectra.runtime.planner")
_ALLOWED_ACTION_KEYS = {"action_id", "tool", "params"}
_ACTION_RESULT_STATUSES = {"ok", "error"}


def _get_registry() -> ToolRegistry:
    registry = get_default_registry()
    registry.discover()
    return registry


def _validate_vector3(value: Any, *, tool_name: str, path: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise PlannerValidationError(
            f"Action '{tool_name}' param '{path}' must be a 3-item JSON array"
        )

    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise PlannerValidationError(
                f"Action '{tool_name}' param '{path}' must contain only numeric values"
            )


def _validate_string(value: Any, *, tool_name: str, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PlannerValidationError(
            f"Action '{tool_name}' param '{path}' must be a non-empty string"
        )


def _validate_ref(
    value: Mapping[str, Any],
    *,
    tool_name: str,
    path: str,
    expected_type: str,
    known_outputs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    if set(value.keys()) != {"$ref"}:
        raise PlannerValidationError(
            f"Action '{tool_name}' has an invalid $ref at {path}"
        )

    raw_ref = value["$ref"]
    if not isinstance(raw_ref, str):
        raise PlannerValidationError(
            f"Action '{tool_name}' has an invalid $ref at {path}"
        )

    ref_parts = raw_ref.split(".")
    if len(ref_parts) != 2 or not ref_parts[0] or not ref_parts[1]:
        raise PlannerValidationError(
            f"Action '{tool_name}' has an invalid $ref at {path}"
        )

    ref_action_id, output_key = ref_parts
    if ref_action_id not in known_outputs:
        raise PlannerValidationError(
            f"Action '{tool_name}' references unknown action_id '{ref_action_id}' at {path}"
        )

    output_schema = known_outputs[ref_action_id]
    if output_key not in output_schema:
        raise PlannerValidationError(
            f"Action '{tool_name}' references unknown output '{output_key}' at {path}"
        )

    output_type = output_schema[output_key].get("type")
    if output_type != expected_type:
        raise PlannerValidationError(
            f"Action '{tool_name}' has type-mismatched $ref at {path}: expected {expected_type}, got {output_type}"
        )


def _validate_param_value(
    value: Any,
    *,
    tool_name: str,
    param_name: str,
    param_spec: dict[str, Any],
    known_outputs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    if isinstance(value, Mapping):
        _validate_ref(
            value,
            tool_name=tool_name,
            path=param_name,
            expected_type=str(param_spec.get("type", "")),
            known_outputs=known_outputs,
        )
        return

    expected_type = param_spec.get("type")
    if expected_type == "string":
        if isinstance(value, str) and value.startswith("$ref("):
            raise PlannerValidationError(
                f"Action '{tool_name}' must encode refs as objects at params.{param_name}"
            )
        _validate_string(value, tool_name=tool_name, path=param_name)
    elif expected_type == "vector3":
        _validate_vector3(value, tool_name=tool_name, path=param_name)
    else:
        raise PlannerValidationError(
            f"Action '{tool_name}' uses unsupported schema type '{expected_type}' for '{param_name}'"
        )

    enum_values = param_spec.get("enum")
    if enum_values is not None and value not in enum_values:
        raise PlannerValidationError(
            f"Action '{tool_name}' param '{param_name}' must be one of {enum_values}"
        )


def _validate_actions(actions: list[dict[str, Any]], registry: ToolRegistry) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        raise PlannerValidationError("Planner output must be a list")

    validated_actions: list[dict[str, Any]] = []
    seen_action_ids: set[str] = set()
    known_outputs: dict[str, dict[str, dict[str, Any]]] = {}

    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, Mapping):
            raise PlannerValidationError(f"Action at index {index} must be an object")

        unexpected_keys = sorted(set(raw_action) - _ALLOWED_ACTION_KEYS)
        if unexpected_keys:
            raise PlannerValidationError(
                f"Action at index {index} has unknown keys: {unexpected_keys}"
            )

        tool_name = raw_action.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise PlannerValidationError(f"Action at index {index} must include a non-empty tool")

        params = raw_action.get("params")
        if not isinstance(params, Mapping):
            raise PlannerValidationError(f"Action '{tool_name}' must include params as an object")

        try:
            tool = registry.get(tool_name)
        except ToolRegistryError as exc:
            raise PlannerValidationError(f"Unknown tool '{tool_name}'") from exc

        action_id = raw_action.get("action_id")
        if action_id is not None:
            if not isinstance(action_id, str) or not action_id.strip():
                raise PlannerValidationError(f"Action '{tool_name}' has an invalid action_id")
            if action_id in seen_action_ids:
                raise PlannerValidationError(f"Duplicate action_id '{action_id}'")
            seen_action_ids.add(action_id)

        allowed_param_names = set(tool.input_schema)
        unexpected_params = sorted(set(params) - allowed_param_names)
        if unexpected_params:
            raise PlannerValidationError(
                f"Action '{tool_name}' has unknown params: {unexpected_params}"
            )

        missing_required = sorted(
            param_name
            for param_name, param_spec in tool.input_schema.items()
            if isinstance(param_spec, dict) and param_spec.get("required") and param_name not in params
        )
        if missing_required:
            raise PlannerValidationError(
                f"Action '{tool_name}' is missing required params: {missing_required}"
            )

        for param_name, raw_value in params.items():
            param_spec = tool.input_schema[param_name]
            if not isinstance(param_spec, dict):
                raise PlannerValidationError(
                    f"Action '{tool_name}' uses invalid schema metadata for '{param_name}'"
                )
            _validate_param_value(
                raw_value,
                tool_name=tool_name,
                param_name=param_name,
                param_spec=param_spec,
                known_outputs=known_outputs,
            )

        validated_action = {
            "tool": tool_name,
            "params": dict(params),
        }
        if action_id is not None:
            validated_action["action_id"] = action_id
            known_outputs[action_id] = tool.output_schema
        validated_actions.append(validated_action)

    return validated_actions


def plan(prompt: str, scene_state: dict[str, Any]) -> PlannerResult:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return PlannerResult(
            status="error",
            actions=[],
            message="No actions returned: empty prompt",
        )

    registry = _get_registry()
    log_structured(_PLANNER_LOGGER, "planner_prompt", {"prompt": normalized_prompt})
    log_structured(_PLANNER_LOGGER, "planner_scene_state", scene_state)
    try:
        actions = generate_actions(normalized_prompt, scene_state)
        validated_actions = _validate_actions(actions, registry)
        log_structured(_PLANNER_LOGGER, "planner_validated_actions", validated_actions)
    except (LLMClientError, PlannerValidationError) as exc:
        message = f"No actions returned: {exc}"
        log_structured(_PLANNER_LOGGER, "planner_error", {"message": message}, level="error")
        return PlannerResult(status="error", actions=[], message=message)

    if not validated_actions:
        message = "No actions returned: planner returned an empty action list"
        log_structured(_PLANNER_LOGGER, "planner_error", {"message": message}, level="error")
        return PlannerResult(status="error", actions=[], message=message)

    result = PlannerResult(status="ok", actions=validated_actions, message="planned")
    if result.status not in _ACTION_RESULT_STATUSES:  # pragma: no cover - defensive guard
        raise PlannerValidationError(f"Invalid planner result status '{result.status}'")
    return result
