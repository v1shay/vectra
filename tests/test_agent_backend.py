from __future__ import annotations

from fastapi.testclient import TestClient

import agent_runtime.main as runtime_main
from agent_runtime.models import AgentStepResponse, AssumptionModel, ExecutionPayloadModel

client = TestClient(runtime_main.app)


def test_agent_step_returns_tool_execution_for_vectra_dev(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main.agent_service,
        "step",
        lambda request: AgentStepResponse(
            status="ok",
            message="Prepared the next step.",
            narration="Creating the first cube.",
            understanding="The Director chose a bounded atomic action.",
            plan=["use mesh.create_primitive as the next step"],
            intended_actions=["mesh.create_primitive({'type': 'cube'})"],
            assumptions=[
                AssumptionModel(
                    key="location",
                    value=[0.0, 0.0, 0.0],
                    reason="Used the world origin as the safest starting point.",
                )
            ],
            preferred_execution_mode="vectra-dev",
            continue_loop=True,
            expected_outcome="The first cube exists.",
            execution=ExecutionPayloadModel(
                kind="tool_actions",
                actions=[
                    {
                        "action_id": "director_1_create",
                        "tool": "mesh.create_primitive",
                        "params": {"type": "cube", "location": [0.0, 0.0, 0.0]},
                    }
                ],
                metadata={"provider": "openai-director", "model": "gpt-5.1"},
            ),
        ),
    )

    response = client.post(
        "/agent/step",
        json={
            "prompt": "make two cubes",
            "scene_state": {"objects": [], "selected_objects": [], "active_object": None},
            "history": [],
            "iteration": 1,
            "execution_mode": "vectra-dev",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["preferred_execution_mode"] == "vectra-dev"
    assert payload["execution"]["kind"] == "tool_actions"
    assert payload["continue_loop"] is True
    assert payload["execution"]["actions"][0]["tool"] == "mesh.create_primitive"
    assert payload["assumptions"][0]["key"] == "location"


def test_agent_step_returns_console_code_for_vectra_code(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main.agent_service,
        "step",
        lambda request: AgentStepResponse(
            status="ok",
            message="Prepared the next step.",
            narration="Using bounded Python because the atomic tool surface is not enough for this step.",
            understanding="The Director explicitly chose vectra-code mode.",
            plan=["use python.execute_blender_snippet as the next step"],
            intended_actions=["python.execute_blender_snippet({'code': 'print(1)'})"],
            assumptions=[],
            preferred_execution_mode="vectra-code",
            continue_loop=True,
            expected_outcome="The corrective snippet runs.",
            execution=ExecutionPayloadModel(
                kind="console_code",
                code="print('vectra-code')",
                metadata={"provider": "openai-director", "model": "gpt-5.1"},
            ),
        ),
    )

    response = client.post(
        "/agent/step",
        json={
            "prompt": "make two cubes",
            "scene_state": {"objects": [], "selected_objects": [], "active_object": None},
            "history": [],
            "iteration": 1,
            "execution_mode": "vectra-code",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["preferred_execution_mode"] == "vectra-code"
    assert payload["execution"]["kind"] == "console_code"
    assert "vectra-code" in payload["execution"]["code"]
