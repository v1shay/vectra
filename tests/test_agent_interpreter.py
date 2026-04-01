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


def test_interpreter_prefers_compiled_actions_from_reasoning_metadata() -> None:
    reasoning = ReasoningStep(
        status="ok",
        narration="Creating a cube.",
        understanding="The user wants a cube.",
        plan=["Create one cube."],
        intended_actions=[],
        expected_outcome="One cube exists.",
        preferred_execution_mode="vectra-dev",
        continue_loop=False,
        metadata={
            "compiled_actions": [
                {
                    "action_id": "create_cube_1",
                    "tool": "mesh.create_primitive",
                    "params": {
                        "primitive_type": "cube",
                        "name": "Cube_1",
                        "location": [0.0, 0.0, 0.0],
                    },
                }
            ],
            "execution_metadata": {
                "affected_logical_ids": ["cube_pair_1"],
                "affected_group_ids": ["group_cube_pair"],
            },
        },
    )

    instruction = interpret_reasoning(reasoning, _context())

    assert instruction.kind == "tool_actions"
    assert instruction.actions[0]["tool"] == "mesh.create_primitive"
    assert instruction.metadata == {
        "affected_logical_ids": ["cube_pair_1"],
        "affected_group_ids": ["group_cube_pair"],
    }


def test_interpreter_builds_console_code_from_compiled_actions_with_refs() -> None:
    reasoning = ReasoningStep(
        status="ok",
        narration="Create and move the cube.",
        understanding="A shared planner produced a two-step plan.",
        plan=["Create a cube.", "Move it."],
        intended_actions=[],
        expected_outcome="The cube exists at the new location.",
        preferred_execution_mode="vectra-code",
        continue_loop=False,
        metadata={
            "compiled_actions": [
                {
                    "action_id": "create_cube",
                    "tool": "mesh.create_primitive",
                    "params": {
                        "primitive_type": "cube",
                        "name": "Cube_1",
                        "location": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "action_id": "move_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube_1",
                        "location": [2.0, 0.0, 0.0],
                    },
                },
            ]
        },
    )

    instruction = interpret_reasoning(reasoning, _context(execution_mode="vectra-code"))

    assert instruction.kind == "console_code"
    assert "primitive_cube_add" in instruction.code
    assert "bpy.data.objects.get('Cube_1')" in instruction.code
