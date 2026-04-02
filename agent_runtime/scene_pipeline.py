from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from .construction import CompiledConstructionPlan, ConstructionError, compile_construction_plan
    from .llm_client import LLMClientError, extract_scene_intent
    from .scene_intent import SceneIntent, normalize_scene_intent
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from agent_runtime.construction import CompiledConstructionPlan, ConstructionError, compile_construction_plan
    from agent_runtime.llm_client import LLMClientError, extract_scene_intent
    from agent_runtime.scene_intent import SceneIntent, normalize_scene_intent

_MINIMUM_SCENE_INTENT_CONFIDENCE = 0.35


@dataclass(frozen=True)
class ScenePipelineResult:
    status: str
    message: str
    prompt: str
    scene_intent: SceneIntent | None = None
    compiled_plan: CompiledConstructionPlan | None = None
    plan_lines: list[str] = field(default_factory=list)
    intended_actions: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    error: str | None = None


def _describe_step(step_kind: str, details: str) -> str:
    return f"{step_kind.replace('_', ' ')}: {details}"


def _describe_actions(actions: list[dict[str, Any]]) -> list[str]:
    descriptions: list[str] = []
    for action in actions:
        tool = action.get("tool")
        params = action.get("params", {})
        if tool == "mesh.create_primitive":
            primitive_type = params.get("primitive_type", "object")
            name = params.get("name", "Object")
            location = params.get("location", [0.0, 0.0, 0.0])
            descriptions.append(
                f"create {primitive_type} {name} at ({float(location[0]):.1f}, {float(location[1]):.1f}, {float(location[2]):.1f})"
            )
            continue
        if tool == "object.transform":
            object_name = params.get("object_name", "Object")
            location = params.get("location")
            if isinstance(location, list) and len(location) == 3:
                descriptions.append(
                    f"move {object_name} to ({float(location[0]):.1f}, {float(location[1]):.1f}, {float(location[2]):.1f})"
                )
                continue
        descriptions.append(str(action))
    return descriptions


def _build_plan_lines(intent: SceneIntent, compiled_plan: CompiledConstructionPlan) -> list[str]:
    plan_lines: list[str] = []
    for entity in intent.entities:
        plan_lines.append(_describe_step("ensure_entity", f"{entity.kind} x{entity.quantity} ({entity.logical_id})"))
    for relationship in intent.relationships:
        target = relationship.target_id or relationship.target_group_id or "reference"
        plan_lines.append(
            _describe_step(
                "satisfy_relation",
                f"{relationship.relation_type} from {relationship.source_id} to {target}",
            )
        )
    if not plan_lines:
        for step in compiled_plan.steps:
            target = step.entity_id or step.relationship_id or step.group_id or step.logical_id
            plan_lines.append(_describe_step(step.kind, target))
    return plan_lines


def _build_success_message(compiled_plan: CompiledConstructionPlan) -> str:
    if compiled_plan.continue_loop:
        return "Planned the next bounded construction step."
    return "Scene intent satisfied."


def build_scene_pipeline(
    prompt: str,
    scene_state: dict[str, Any],
    *,
    max_construction_steps: int | None,
) -> ScenePipelineResult:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return ScenePipelineResult(
            status="error",
            message="Empty prompt",
            prompt=normalized_prompt,
            error="Empty prompt",
        )

    try:
        raw_intent = extract_scene_intent(normalized_prompt, scene_state)
        normalized_intent = normalize_scene_intent(
            raw_intent,
            minimum_confidence=_MINIMUM_SCENE_INTENT_CONFIDENCE,
        )
        if normalized_intent.status != "ok":
            message = normalized_intent.reasoning or "The scene intent could not be grounded safely."
            return ScenePipelineResult(
                status="error",
                message=message,
                prompt=normalized_prompt,
                scene_intent=normalized_intent,
                uncertainty_notes=list(normalized_intent.uncertainty_notes),
                error=message,
            )

        compiled_plan = compile_construction_plan(
            normalized_intent,
            scene_state=scene_state,
            max_construction_steps=max_construction_steps,
        )
    except (LLMClientError, ConstructionError) as exc:
        message = str(exc)
        return ScenePipelineResult(
            status="error",
            message=message,
            prompt=normalized_prompt,
            error=message,
        )

    intended_actions = _describe_actions(compiled_plan.actions)
    plan_lines = _build_plan_lines(normalized_intent, compiled_plan)
    if not compiled_plan.actions:
        status = "complete" if not compiled_plan.continue_loop else "error"
        message = (
            "The requested scene is already satisfied."
            if status == "complete"
            else "The scene still has remaining work, but no executable actions were produced."
        )
        return ScenePipelineResult(
            status=status,
            message=message,
            prompt=normalized_prompt,
            scene_intent=normalized_intent,
            compiled_plan=compiled_plan,
            plan_lines=plan_lines,
            intended_actions=intended_actions,
            uncertainty_notes=list(normalized_intent.uncertainty_notes),
            error=None if status == "complete" else message,
        )

    return ScenePipelineResult(
        status="ok",
        message=_build_success_message(compiled_plan),
        prompt=normalized_prompt,
        scene_intent=normalized_intent,
        compiled_plan=compiled_plan,
        plan_lines=plan_lines,
        intended_actions=intended_actions,
        uncertainty_notes=list(normalized_intent.uncertainty_notes),
    )
