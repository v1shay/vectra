from __future__ import annotations

import pytest

from agent_runtime.agent.reasoner import reason_step
from agent_runtime.agent.service import AgentService
from agent_runtime.memory.manager import MemoryManager
from agent_runtime.memory.providers.null import NullMemoryProvider
from agent_runtime.models import AgentStepRequest
from agent_runtime.planner import plan
from agent_runtime.scene_intent import SceneEntity, SceneIntent, SceneRelationship, SceneTransformIntent


def _shared_intent_for_prompt(prompt: str, scene_state: dict[str, object]) -> SceneIntent:
    del scene_state
    lowered = prompt.strip().lower()
    if lowered == "make 2 cubes and move them apart":
        return SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create two cubes and spread them.",
            entities=[SceneEntity(logical_id="cube_pair", kind="cube", quantity=2, group_id="cube_pair_group")],
            relationships=[
                SceneRelationship(
                    logical_id="spread_pair",
                    relation_type="relative_offset",
                    source_id="cube_pair",
                    offset=[3.0, 0.0, 0.0],
                )
            ],
        )
    if lowered == "make a plane down 10 right 3 and a cube left 17 down 4":
        return SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create two entities with fixed offsets.",
            entities=[
                SceneEntity(
                    logical_id="plane_left",
                    kind="plane",
                    initial_transform=SceneTransformIntent(offset=[3.0, 0.0, -10.0]),
                ),
                SceneEntity(
                    logical_id="cube_right",
                    kind="cube",
                    initial_transform=SceneTransformIntent(offset=[-17.0, 0.0, -4.0]),
                ),
            ],
        )
    if lowered == "put a cube on top of another":
        return SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Stack two cubes.",
            entities=[
                SceneEntity(logical_id="base_cube", kind="cube"),
                SceneEntity(logical_id="top_cube", kind="cube"),
            ],
            relationships=[
                SceneRelationship(
                    logical_id="stack_top",
                    relation_type="above",
                    source_id="top_cube",
                    target_id="base_cube",
                    metadata={"touching": True},
                )
            ],
        )
    if lowered == "make a staircase of cubes":
        return SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create a staircase pattern.",
            entities=[
                SceneEntity(
                    logical_id="stairs",
                    kind="cube",
                    quantity=5,
                    group_id="stairs_group",
                )
            ],
            metadata={"pattern": "staircase", "stair_offset": [1.25, 0.0, 1.0]},
        )
    raise AssertionError(f"Unexpected prompt in shared test fixture: {prompt}")


def test_task_create_and_agent_reasoner_both_use_shared_compile_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _shared_intent_for_prompt)
    calls: list[int] = []

    original_compile = __import__("agent_runtime.scene_pipeline", fromlist=["compile_construction_plan"]).compile_construction_plan

    def tracked_compile(*args, **kwargs):
        calls.append(1)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr("agent_runtime.scene_pipeline.compile_construction_plan", tracked_compile)

    task_result = plan("make 2 cubes and move them apart", {"objects": []})
    reasoning = reason_step(
        __import__("agent_runtime.agent.models", fromlist=["AgentContext"]).AgentContext(
            user_prompt="make 2 cubes and move them apart",
            scene_state={"objects": [], "selected_objects": [], "active_object": None},
            screenshot=None,
            history=[],
            iteration=1,
            execution_mode="vectra-dev",
            memory_results=[],
        )
    )

    assert task_result.status == "ok"
    assert reasoning.status == "ok"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "prompt",
    [
        "make 2 cubes and move them apart",
        "make a plane down 10 right 3 and a cube left 17 down 4",
        "put a cube on top of another",
        "make a staircase of cubes",
    ],
)
def test_task_create_acceptance_prompts(monkeypatch: pytest.MonkeyPatch, prompt: str) -> None:
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _shared_intent_for_prompt)

    result = plan(prompt, {"objects": [], "selected_objects": [], "active_object": None})

    assert result.status == "ok"
    assert result.actions


@pytest.mark.parametrize("execution_mode", ["vectra-dev", "vectra-code"])
@pytest.mark.parametrize(
    "prompt",
    [
        "make 2 cubes and move them apart",
        "make a plane down 10 right 3 and a cube left 17 down 4",
        "put a cube on top of another",
        "make a staircase of cubes",
    ],
)
def test_agent_step_acceptance_prompts(
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    execution_mode: str,
) -> None:
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _shared_intent_for_prompt)
    service = AgentService(memory_manager=MemoryManager(provider=NullMemoryProvider()))

    response = service.step(
        AgentStepRequest(
            prompt=prompt,
            scene_state={"objects": [], "selected_objects": [], "active_object": None},
            history=[],
            iteration=1,
            execution_mode=execution_mode,  # type: ignore[arg-type]
        )
    )

    assert response.status in {"ok", "complete"}
    assert response.execution.kind in {"tool_actions", "console_code"}
    if execution_mode == "vectra-code":
        assert response.execution.kind == "console_code"
        assert response.execution.code
    else:
        assert response.execution.kind == "tool_actions"
        assert response.execution.actions
