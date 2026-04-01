from __future__ import annotations

from typing import Any

from agent_runtime.scene_pipeline import build_scene_pipeline

from .models import AgentContext, ExecutionMode, ReasoningStep


def _make_reasoning(
    *,
    narration: str,
    understanding: str,
    plan: list[str],
    intended_actions: list[str],
    expected_outcome: str,
    preferred_execution_mode: ExecutionMode,
    continue_loop: bool,
    status: str = "ok",
    question: str | None = None,
    error: str | None = None,
    uncertainty_notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReasoningStep:
    return ReasoningStep(
        status=status,  # type: ignore[arg-type]
        narration=narration,
        understanding=understanding,
        plan=plan,
        intended_actions=intended_actions,
        expected_outcome=expected_outcome,
        preferred_execution_mode=preferred_execution_mode,
        continue_loop=continue_loop,
        question=question,
        error=error,
        uncertainty_notes=uncertainty_notes or [],
        metadata=metadata or {},
    )


def reason_step(context: AgentContext) -> ReasoningStep:
    pipeline_result = build_scene_pipeline(
        context.user_prompt,
        context.scene_state,
        max_construction_steps=1,
    )
    compiled_plan = pipeline_result.compiled_plan
    scene_intent = pipeline_result.scene_intent

    understanding = (
        scene_intent.reasoning
        if scene_intent is not None and scene_intent.reasoning.strip()
        else "The agent is using the shared SceneIntent construction pipeline."
    )
    expected_outcome = (
        compiled_plan.expected_outcome
        if compiled_plan is not None
        else pipeline_result.message
    )
    metadata = {
        "compiled_actions": compiled_plan.actions if compiled_plan is not None else [],
        "execution_metadata": {
            "affected_logical_ids": compiled_plan.affected_logical_ids if compiled_plan is not None else [],
            "affected_group_ids": compiled_plan.affected_group_ids if compiled_plan is not None else [],
        },
    }

    if pipeline_result.status == "error":
        return _make_reasoning(
            narration=pipeline_result.message,
            understanding=understanding,
            plan=pipeline_result.plan_lines or ["Extract scene intent.", "Stop safely when grounding fails."],
            intended_actions=[],
            expected_outcome=expected_outcome,
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="error",
            error=pipeline_result.error or pipeline_result.message,
            uncertainty_notes=list(pipeline_result.uncertainty_notes),
            metadata=metadata,
        )

    if pipeline_result.status == "complete":
        return _make_reasoning(
            narration=pipeline_result.message,
            understanding=understanding,
            plan=pipeline_result.plan_lines,
            intended_actions=[],
            expected_outcome=expected_outcome,
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="complete",
            uncertainty_notes=list(pipeline_result.uncertainty_notes),
            metadata=metadata,
        )

    return _make_reasoning(
        narration=pipeline_result.message,
        understanding=understanding,
        plan=pipeline_result.plan_lines,
        intended_actions=pipeline_result.intended_actions,
        expected_outcome=expected_outcome,
        preferred_execution_mode=context.execution_mode,
        continue_loop=bool(compiled_plan and compiled_plan.continue_loop),
        uncertainty_notes=list(pipeline_result.uncertainty_notes),
        metadata=metadata,
    )
