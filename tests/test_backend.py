from __future__ import annotations

from fastapi.testclient import TestClient

from agent_runtime.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_task_returns_stub_response() -> None:
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
        "message": "received",
        "actions": [],
    }


def test_create_task_is_deterministic_for_identical_input() -> None:
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
        "message": "received",
        "actions": [],
    }
