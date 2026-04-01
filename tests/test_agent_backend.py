from __future__ import annotations

from fastapi.testclient import TestClient

import agent_runtime.main as runtime_main

client = TestClient(runtime_main.app)


def test_agent_step_returns_tool_execution_for_vectra_dev() -> None:
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


def test_agent_step_returns_console_code_for_vectra_code() -> None:
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
    assert "primitive_cube_add" in payload["execution"]["code"]
