from __future__ import annotations

from agent_runtime.agent.models import AgentContext
from agent_runtime.agent.reasoner import reason_step
from agent_runtime.director.models import AssumptionRecord, DirectorTurn


def _context(prompt: str, *, execution_mode: str = "vectra-dev") -> AgentContext:
    return AgentContext(
        user_prompt=prompt,
        scene_state={"objects": [], "selected_objects": [], "active_object": None},
        screenshot=None,
        history=[],
        iteration=1,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        memory_results=[],
    )


def test_reasoner_uses_director_loop_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.agent.reasoner._DIRECTOR_LOOP.step",
        lambda context: DirectorTurn(
            status="ok",
            message="Prepared the next step.",
            narration="I am creating the first cube.",
            understanding="The Director chose a bounded action.",
            plan=["use mesh.create_primitive as the next step"],
            intended_actions=["mesh.create_primitive({'type': 'cube'})"],
            expected_outcome="The first cube exists.",
            continue_loop=True,
            assumptions=[
                AssumptionRecord(
                    key="location",
                    value=[0.0, 0.0, 0.0],
                    reason="Used the world origin as the safest starting point.",
                )
            ],
            metadata={"actions": [{"tool": "mesh.create_primitive", "params": {"type": "cube"}}]},
        ),
    )

    reasoning = reason_step(_context("create 6 cubes"))

    assert reasoning.status == "ok"
    assert reasoning.plan == ["use mesh.create_primitive as the next step"]
    assert reasoning.assumptions[0]["key"] == "location"
    assert reasoning.metadata["actions"][0]["tool"] == "mesh.create_primitive"


def test_reasoner_surfaces_director_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.agent.reasoner._DIRECTOR_LOOP.step",
        lambda context: DirectorTurn(
            status="error",
            message="No provider was available.",
            narration="No provider was available.",
            understanding="The Director could not get a usable answer from any configured provider.",
            plan=["surface the provider failure"],
            intended_actions=[],
            expected_outcome="No execution could be prepared.",
            continue_loop=False,
            error="No provider was available.",
        ),
    )

    reasoning = reason_step(_context("make a cool scene"))

    assert reasoning.status == "error"
    assert reasoning.error == "No provider was available."
    assert reasoning.intended_actions == []
