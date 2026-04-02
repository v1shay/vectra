from __future__ import annotations

from agent_runtime.director.loop import DirectorLoop

from .models import AgentContext, ExecutionMode, ReasoningStep

_DIRECTOR_LOOP = DirectorLoop()


def _make_reasoning(
    *,
    narration: str,
    understanding: str,
    plan: list[str],
    intended_actions: list[str],
    expected_outcome: str,
    preferred_execution_mode: ExecutionMode,
    continue_loop: bool,
    assumptions: list[dict[str, object]] | None = None,
    status: str = "ok",
    question: str | None = None,
    error: str | None = None,
    uncertainty_notes: list[str] | None = None,
    metadata: dict[str, object] | None = None,
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
        assumptions=assumptions or [],
        question=question,
        error=error,
        uncertainty_notes=uncertainty_notes or [],
        metadata=metadata or {},
    )


def reason_step(context: AgentContext) -> ReasoningStep:
    turn = _DIRECTOR_LOOP.step(
        __import__("agent_runtime.director.models", fromlist=["DirectorContext"]).DirectorContext(
            user_prompt=context.user_prompt,
            scene_state=context.scene_state,
            screenshot=context.screenshot,
            history=context.history,
            iteration=context.iteration,
            execution_mode=context.execution_mode,
            memory_results=context.memory_results,
        )
    )
    assumptions = [
        {"key": item.key, "value": item.value, "reason": item.reason}
        for item in turn.assumptions
    ]
    return _make_reasoning(
        narration=turn.narration,
        understanding=turn.understanding,
        plan=list(turn.plan),
        intended_actions=list(turn.intended_actions),
        expected_outcome=turn.expected_outcome,
        preferred_execution_mode=context.execution_mode,
        continue_loop=turn.continue_loop,
        assumptions=assumptions,
        status=turn.status,
        question=turn.question,
        error=turn.error,
        metadata=dict(turn.metadata),
    )
