from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vectra.tools.registry import ToolRegistry, ToolRegistryError, get_default_registry

try:
    from .llm_client import LLMClientError, generate_actions
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from llm_client import LLMClientError, generate_actions


class PlannerValidationError(Exception):
    """Raised when planner actions fail structural validation."""


@dataclass(frozen=True)
class PlannerResult:
    actions: list[dict[str, Any]]
    message: str


def _get_registry() -> ToolRegistry:
    registry = get_default_registry()
    registry.discover()
    return registry


def _validate_actions(actions: list[dict[str, Any]], registry: ToolRegistry) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        raise PlannerValidationError("Planner output must be a list")

    validated_actions: list[dict[str, Any]] = []
    seen_action_ids: set[str] = set()

    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, Mapping):
            raise PlannerValidationError(f"Action at index {index} must be an object")

        tool_name = raw_action.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise PlannerValidationError(f"Action at index {index} must include a non-empty tool")

        params = raw_action.get("params")
        if not isinstance(params, Mapping):
            raise PlannerValidationError(f"Action '{tool_name}' must include params as an object")

        try:
            registry.get(tool_name)
        except ToolRegistryError as exc:
            raise PlannerValidationError(f"Unknown tool '{tool_name}'") from exc

        action_id = raw_action.get("action_id")
        if action_id is not None:
            if not isinstance(action_id, str) or not action_id.strip():
                raise PlannerValidationError(f"Action '{tool_name}' has an invalid action_id")
            if action_id in seen_action_ids:
                raise PlannerValidationError(f"Duplicate action_id '{action_id}'")
            seen_action_ids.add(action_id)

        validated_action = {
            "tool": tool_name,
            "params": dict(params),
        }
        if action_id is not None:
            validated_action["action_id"] = action_id
        validated_actions.append(validated_action)

    return validated_actions


def plan(prompt: str, scene_state: dict[str, Any]) -> PlannerResult:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return PlannerResult(actions=[], message="No actions returned: empty prompt")

    registry = _get_registry()
    try:
        actions = generate_actions(normalized_prompt, scene_state)
        validated_actions = _validate_actions(actions, registry)
    except (LLMClientError, PlannerValidationError) as exc:
        return PlannerResult(actions=[], message=f"No actions returned: {exc}")

    return PlannerResult(actions=validated_actions, message="planned")
