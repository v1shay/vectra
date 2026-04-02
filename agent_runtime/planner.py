from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vectra.utils.logging import get_vectra_logger, log_structured

from .director.loop import DirectorLoop
from .director.models import DirectorContext


@dataclass(frozen=True)
class PlannerResult:
    status: str
    actions: list[dict[str, Any]]
    message: str
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_PLANNER_LOGGER = get_vectra_logger("vectra.runtime.planner")
_DIRECTOR_LOOP = DirectorLoop()


def plan(prompt: str, scene_state: dict[str, Any]) -> PlannerResult:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return PlannerResult(
            status="error",
            actions=[],
            message="No actions returned: empty prompt",
        )

    log_structured(_PLANNER_LOGGER, "planner_prompt", {"prompt": normalized_prompt})
    log_structured(_PLANNER_LOGGER, "planner_scene_state", scene_state)
    turn = _DIRECTOR_LOOP.step(
        DirectorContext(
            user_prompt=normalized_prompt,
            scene_state=scene_state,
            screenshot=None,
            history=[],
            iteration=1,
            execution_mode="vectra-dev",
            memory_results=[],
        )
    )
    assumptions = [
        {"key": item.key, "value": item.value, "reason": item.reason}
        for item in turn.assumptions
    ]
    actions = turn.metadata.get("actions", []) if isinstance(turn.metadata, dict) else []
    if turn.status == "ok" and isinstance(actions, list) and actions:
        return PlannerResult(
            status="ok",
            actions=[action for action in actions if isinstance(action, dict)],
            message=turn.message or "Prepared the next executable step.",
            assumptions=assumptions,
            metadata=dict(turn.metadata),
        )

    if turn.status == "complete":
        return PlannerResult(
            status="ok",
            actions=[],
            message=turn.message or "Task complete.",
            assumptions=assumptions,
            metadata=dict(turn.metadata),
        )

    if turn.status == "clarify":
        return PlannerResult(
            status="error",
            actions=[],
            message=turn.question or turn.message or "Clarification required.",
            assumptions=assumptions,
            metadata=dict(turn.metadata),
        )

    return PlannerResult(
        status="error",
        actions=[],
        message=turn.error or turn.message or "No actions returned.",
        assumptions=assumptions,
        metadata=dict(turn.metadata),
    )
