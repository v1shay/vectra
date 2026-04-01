from __future__ import annotations

from fastapi.testclient import TestClient

import agent_runtime.main as runtime_main
from agent_runtime.scene_intent import SceneEntity, SceneIntent, SceneRelationship

client = TestClient(runtime_main.app)


def _intent_for_prompt(prompt: str, scene_state: dict[str, object]) -> SceneIntent:
    del scene_state
    lowered = prompt.strip().lower()
    if "apart" in lowered:
        return SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create two cubes and separate them.",
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
    return SceneIntent(
        status="ok",
        confidence=0.9,
        reasoning="Create two cubes.",
        entities=[SceneEntity(logical_id="cube_pair", kind="cube", quantity=2, group_id="cube_pair_group")],
    )


def test_agent_step_returns_tool_execution_for_vectra_dev(monkeypatch) -> None:
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _intent_for_prompt)

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
    assert payload["execution"]["metadata"]["affected_logical_ids"] == ["cube_pair_1"]


def test_agent_step_returns_console_code_for_vectra_code(monkeypatch) -> None:
    monkeypatch.setattr("agent_runtime.scene_pipeline.extract_scene_intent", _intent_for_prompt)

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
