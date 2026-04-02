from __future__ import annotations

from agent_runtime.agent.service import AgentService
from agent_runtime.director.models import DirectorTurn
from agent_runtime.memory.manager import MemoryManager
from agent_runtime.memory.providers.null import NullMemoryProvider
from agent_runtime.models import AgentStepRequest
from agent_runtime.planner import plan


def test_task_create_and_agent_step_share_the_same_director_loop(monkeypatch) -> None:
    seen_prompts: list[str] = []

    def fake_step(self, context):
        seen_prompts.append(context.user_prompt)
        return DirectorTurn(
            status="ok",
            message="Prepared the next step.",
            narration="Creating the next primitive.",
            understanding="The shared DirectorLoop handled the request.",
            plan=["use mesh.create_primitive as the next step"],
            intended_actions=["mesh.create_primitive({'type': 'cube'})"],
            expected_outcome="The scene progresses.",
            continue_loop=True,
            assumptions=[],
            metadata={
                "actions": [
                    {
                        "action_id": "director_1_create",
                        "tool": "mesh.create_primitive",
                        "params": {"type": "cube", "location": [0.0, 0.0, 0.0]},
                    }
                ]
            },
        )

    monkeypatch.setattr("agent_runtime.planner._DIRECTOR_LOOP.step", lambda context: fake_step(None, context))
    monkeypatch.setattr("agent_runtime.agent.reasoner._DIRECTOR_LOOP.step", lambda context: fake_step(None, context))

    task_result = plan("create 6 cubes", {"objects": []})
    agent_result = AgentService(memory_manager=MemoryManager(provider=NullMemoryProvider())).step(
        AgentStepRequest(
            prompt="create 6 cubes",
            scene_state={"objects": [], "selected_objects": [], "active_object": None},
            history=[],
            iteration=1,
            execution_mode="vectra-dev",
        )
    )

    assert task_result.status == "ok"
    assert agent_result.status == "ok"
    assert seen_prompts == ["create 6 cubes", "create 6 cubes"]
