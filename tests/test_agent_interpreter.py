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


def test_interpreter_translates_natural_language_into_tool_actions() -> None:
    reasoning = ReasoningStep(
        status="ok",
        narration="Creating a cube.",
        understanding="The user wants a cube.",
        plan=["Create one cube."],
        intended_actions=["create a cube named VectraCube1 at (0.0, 0.0, 0.0)"],
        expected_outcome="One cube exists.",
        preferred_execution_mode="vectra-dev",
        continue_loop=False,
    )

    instruction = interpret_reasoning(reasoning, _context())

    assert instruction.kind == "tool_actions"
    assert instruction.actions == [
        {
            "action_id": "agent_create_1",
            "tool": "mesh.create_primitive",
            "params": {
                "primitive_type": "cube",
                "name": "VectraCube1",
                "location": [0.0, 0.0, 0.0],
            },
        }
    ]


def test_interpreter_builds_console_code_from_fallback_actions_with_refs() -> None:
    reasoning = ReasoningStep(
        status="ok",
        narration="Create and move the cube.",
        understanding="A fallback planner produced a two-step plan.",
        plan=["Create a cube.", "Move it."],
        intended_actions=[],
        expected_outcome="The cube exists at the new location.",
        preferred_execution_mode="vectra-code",
        continue_loop=False,
        metadata={
            "fallback_actions": [
                {
                    "action_id": "create_cube",
                    "tool": "mesh.create_primitive",
                    "params": {
                        "primitive_type": "cube",
                        "name": "VectraCube",
                        "location": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "action_id": "move_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": {"$ref": "create_cube.object_name"},
                        "location": [2.0, 0.0, 0.0],
                    },
                },
            ]
        },
    )

    instruction = interpret_reasoning(reasoning, _context(execution_mode="vectra-code"))

    assert instruction.kind == "console_code"
    assert "primitive_cube_add" in instruction.code
    assert "bpy.data.objects.get('VectraCube')" in instruction.code
