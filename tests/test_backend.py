from __future__ import annotations

from fastapi.testclient import TestClient

import agent_runtime.main as runtime_main
from agent_runtime.planner import PlannerResult
from tests.action_fixtures import CREATE_CUBE_ACTIONS

client = TestClient(runtime_main.app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_task_returns_director_wrapper_response(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="ok",
            actions=CREATE_CUBE_ACTIONS,
            message=f"planned for {prompt}:{scene_state.get('current_frame', 'missing')}",
            assumptions=[{"key": "location", "value": [0.0, 0.0, 0.0], "reason": "Used the world origin."}],
            metadata={"provider": "openai-director", "model": "gpt-5.1"},
        ),
    )
    payload = {
        "prompt": "Create a chair",
        "scene_state": {
            "active_object": None,
            "selected_objects": [],
            "current_frame": 1,
        },
        "images": [],
    }

    response = client.post("/task/create", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "planned for Create a chair:1",
        "actions": CREATE_CUBE_ACTIONS,
        "assumptions": [{"key": "location", "value": [0.0, 0.0, 0.0], "reason": "Used the world origin."}],
        "metadata": {"provider": "openai-director", "model": "gpt-5.1"},
    }


def test_create_task_returns_error_status_for_failed_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="error",
            actions=[],
            message="No actions returned: provider unavailable",
            assumptions=[],
            metadata={},
        ),
    )

    response = client.post(
        "/task/create",
        json={
            "prompt": "move cube somewhere weird",
            "scene_state": {},
            "images": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "error",
        "message": "No actions returned: provider unavailable",
        "actions": [],
        "assumptions": [],
        "metadata": {},
    }
