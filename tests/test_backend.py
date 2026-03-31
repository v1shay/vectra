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


def test_create_task_returns_planner_response(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="ok",
            actions=CREATE_CUBE_ACTIONS,
            message=f"planned for {prompt}:{scene_state.get('current_frame', 'missing')}",
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
    }


def test_create_task_is_deterministic_for_identical_input(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="ok",
            actions=CREATE_CUBE_ACTIONS,
            message=f"planned for {prompt}:{scene_state.get('current_frame', 'missing')}",
        ),
    )
    payload = {
        "prompt": "Make it blue",
        "scene_state": {
            "active_object": "Cube",
            "selected_objects": ["Cube"],
            "current_frame": 24,
        },
        "images": [],
    }

    first = client.post("/task/create", json=payload)
    second = client.post("/task/create", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json() == {
        "status": "ok",
        "message": "planned for Make it blue:24",
        "actions": CREATE_CUBE_ACTIONS,
    }


def test_create_task_action_schema_is_structured(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="ok",
            actions=CREATE_CUBE_ACTIONS,
            message=f"planned for {prompt}:{len(scene_state)}",
        ),
    )
    response = client.post(
        "/task/create",
        json={
            "prompt": "Create a cube",
            "scene_state": {},
            "images": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["actions"], list)
    assert payload["actions"][0]["action_id"] == "create_cube"
    assert payload["actions"][0]["tool"] == "mesh.create_primitive"
    assert payload["actions"][0]["params"]["name"] == "VectraCube"


def test_create_task_returns_error_status_for_failed_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="error",
            actions=[],
            message="No actions returned: invalid request",
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
        "message": "No actions returned: invalid request",
        "actions": [],
    }
