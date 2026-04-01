from __future__ import annotations

from agent_runtime.agent.reasoner import reason_step
from agent_runtime.agent.models import AgentContext


def _context(prompt: str, scene_state: dict[str, object], *, iteration: int = 1) -> AgentContext:
    return AgentContext(
        user_prompt=prompt,
        scene_state=scene_state,
        screenshot=None,
        history=[],
        iteration=iteration,
        execution_mode="vectra-dev",
        memory_results=[],
    )


def test_reasoner_reflects_follow_up_iterations_for_staircase() -> None:
    first = reason_step(
        _context(
            "create a staircase of cubes",
            {"objects": [], "selected_objects": [], "active_object": None},
            iteration=1,
        )
    )
    second = reason_step(
        _context(
            "create a staircase of cubes",
            {
                "objects": [
                    {
                        "name": "VectraStair1",
                        "type": "MESH",
                        "location": [0.0, 0.0, 0.0],
                        "dimensions": [2.0, 2.0, 2.0],
                    }
                ],
                "selected_objects": [],
                "active_object": None,
            },
            iteration=2,
        )
    )

    assert "VectraStair1" in first.intended_actions[0]
    assert "VectraStair2" in second.intended_actions[0]
    assert second.continue_loop is True
