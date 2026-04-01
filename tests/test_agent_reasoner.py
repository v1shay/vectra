from __future__ import annotations

from agent_runtime.agent.models import AgentContext
from agent_runtime.agent.reasoner import reason_step
from agent_runtime.construction import CompiledConstructionPlan, ConstructionState
from agent_runtime.scene_intent import SceneEntity, SceneIntent
from agent_runtime.scene_pipeline import ScenePipelineResult


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


def test_reasoner_uses_shared_pipeline_for_phrase_regressions(monkeypatch) -> None:
    prompts = [
        "make 2 cubes and move them apart",
        "put a cube on top of another",
        "make a staircase of cubes",
    ]
    seen_prompts: list[str] = []

    def fake_pipeline(prompt: str, scene_state: dict[str, object], *, max_construction_steps: int | None):
        del scene_state
        seen_prompts.append(prompt)
        assert max_construction_steps == 1
        return ScenePipelineResult(
            status="ok",
            message="Planned the next bounded construction step.",
            prompt=prompt,
            scene_intent=SceneIntent(
                status="ok",
                confidence=0.9,
                reasoning="Shared SceneIntent reasoning.",
                entities=[SceneEntity(logical_id="cube_pair", kind="cube", quantity=2)],
            ),
            compiled_plan=CompiledConstructionPlan(
                actions=[
                    {
                        "action_id": "create_cube_pair_1",
                        "tool": "mesh.create_primitive",
                        "params": {
                            "primitive_type": "cube",
                            "name": "Cube_1",
                            "location": [0.0, 0.0, 0.0],
                        },
                    }
                ],
                state=ConstructionState(),
                steps=[],
                affected_logical_ids=["cube_pair_1"],
                affected_group_ids=["group_cube_pair"],
                continue_loop=True,
                expected_outcome="The scene is partially constructed and has remaining construction steps.",
            ),
            plan_lines=["ensure entity: cube x2 (cube_pair)"],
            intended_actions=["create cube Cube_1 at (0.0, 0.0, 0.0)"],
        )

    monkeypatch.setattr("agent_runtime.agent.reasoner.build_scene_pipeline", fake_pipeline)

    results = [
        reason_step(_context(prompt, {"objects": [], "selected_objects": [], "active_object": None}))
        for prompt in prompts
    ]

    assert seen_prompts == prompts
    assert all(result.plan == ["ensure entity: cube x2 (cube_pair)"] for result in results)
    assert all(result.metadata["compiled_actions"][0]["tool"] == "mesh.create_primitive" for result in results)


def test_reasoner_surfaces_pipeline_errors_structurally(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.agent.reasoner.build_scene_pipeline",
        lambda prompt, scene_state, max_construction_steps=None: ScenePipelineResult(
            status="error",
            message="scene intent parse failed",
            prompt=prompt,
            error="scene intent parse failed",
        ),
    )

    reasoning = reason_step(
        _context("make 2 cubes", {"objects": [], "selected_objects": [], "active_object": None})
    )

    assert reasoning.status == "error"
    assert reasoning.error == "scene intent parse failed"
    assert reasoning.intended_actions == []
