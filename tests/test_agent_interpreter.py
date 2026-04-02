from __future__ import annotations

from agent_runtime.agent.interpreter import interpret_reasoning
from agent_runtime.agent.models import AgentContext, ReasoningStep


def _context(*, execution_mode: str = "vectra-dev") -> AgentContext:
    return AgentContext(
        user_prompt="make two cubes",
        scene_state={"objects": [], "selected_objects": [], "active_object": None},
        screenshot=None,
        history=[],
        iteration=1,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        memory_results=[],
    )


def test_interpreter_uses_actions_from_director_metadata() -> None:
    reasoning = ReasoningStep(
        status="ok",
        narration="Creating the first cube.",
        understanding="The Director chose a bounded tool call.",
        plan=["use mesh.create_primitive as the next step"],
        intended_actions=["mesh.create_primitive({'type': 'cube'})"],
        expected_outcome="The first cube exists.",
        preferred_execution_mode="vectra-dev",
        continue_loop=True,
        assumptions=[],
        metadata={
            "actions": [
                {
                    "action_id": "director_1_create",
                    "tool": "mesh.create_primitive",
                    "params": {"type": "cube", "location": [0.0, 0.0, 0.0]},
                }
            ],
            "provider": "openai-director",
        },
    )

    instruction = interpret_reasoning(reasoning, _context())

    assert instruction.kind == "tool_actions"
    assert instruction.actions[0]["tool"] == "mesh.create_primitive"
    assert instruction.metadata["provider"] == "openai-director"


def test_interpreter_returns_console_code_for_vectra_code() -> None:
    reasoning = ReasoningStep(
        status="ok",
        narration="Using bounded Python because the atomic tool surface cannot express this step yet.",
        understanding="The Director chose the explicit vectra-code escape hatch.",
        plan=["use python.execute_blender_snippet as the next step"],
        intended_actions=["python.execute_blender_snippet({'code': 'print(1)'})"],
        expected_outcome="The corrective code runs.",
        preferred_execution_mode="vectra-code",
        continue_loop=True,
        assumptions=[],
        metadata={"code": "print('vectra-code')"},
    )

    instruction = interpret_reasoning(reasoning, _context(execution_mode="vectra-code"))

    assert instruction.kind == "console_code"
    assert "vectra-code" in instruction.code


def test_interpreter_returns_no_execution_for_completion() -> None:
    reasoning = ReasoningStep(
        status="complete",
        narration="The scene looks good now.",
        understanding="The Director finished the task.",
        plan=["finish the task"],
        intended_actions=[],
        expected_outcome="The task is complete.",
        preferred_execution_mode="vectra-dev",
        continue_loop=False,
        assumptions=[],
    )

    instruction = interpret_reasoning(reasoning, _context())

    assert instruction.kind == "none"
