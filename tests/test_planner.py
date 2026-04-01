from __future__ import annotations

import httpx
import pytest

import agent_runtime.llm_client as llm_client_module
import agent_runtime.planner as planner_module
from agent_runtime.construction import CompiledConstructionPlan, ConstructionState
from agent_runtime.llm_client import (
    LLMRequestError,
    LLMTimeoutError,
    extract_scene_intent,
)
from agent_runtime.planner import PlannerResult, plan
from agent_runtime.scene_intent import SceneEntity, SceneIntent
from agent_runtime.scene_pipeline import ScenePipelineResult


class FakeLLMResponse:
    def __init__(self, content: str, status_code: int = 200) -> None:
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("POST", "http://testserver/chat/completions"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": self._content,
                    }
                }
            ]
        }


@pytest.fixture(autouse=True)
def _disable_ollama_discovery_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module, "_read_ollama_config", lambda: None)


def _configure_primary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_LLM_BASE_URL", "https://api.mistral.ai/v1")
    monkeypatch.setenv("VECTRA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("VECTRA_LLM_MODEL", "mistral-medium-latest")
    monkeypatch.setenv("VECTRA_LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("VECTRA_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("VECTRA_LLM_SCENE_OBJECT_LIMIT", "10")


def test_system_prompt_uses_scene_intent_schema_without_tool_catalog() -> None:
    prompt = llm_client_module._system_prompt()

    assert "SceneIntent schema" in prompt
    assert "Do not emit tool calls." in prompt
    assert "Supported tool catalog" not in prompt


def test_user_content_uses_compact_scene_summary_without_raw_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)
    settings = llm_client_module.read_runtime_settings()

    content = llm_client_module._user_content(
        "slide it over",
        {
            "active_object": "Cube",
            "selected_objects": ["Cube"],
            "objects": [
                {
                    "name": "Cube",
                    "type": "MESH",
                    "selected": True,
                    "active": True,
                    "location": [1.0, 2.0, 3.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            ],
        },
        settings=settings,
    )

    assert "Compact object list:" in content
    assert "Raw scene_state JSON" not in content
    assert "name=Cube" in content


def test_extract_scene_intent_retries_timeout_exactly_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)
    attempts: list[int] = []

    def fake_post(*args, **kwargs):
        del args, kwargs
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("timed out")
        return FakeLLMResponse(
            '{"status":"ok","confidence":0.9,"reasoning":"test","entities":[{"logical_id":"cube_pair","kind":"cube","quantity":2}]}'
        )

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)

    intent = extract_scene_intent("make 2 cubes", {})

    assert intent.status == "ok"
    assert attempts == [1, 2, 3]


def test_extract_scene_intent_does_not_retry_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)
    calls: list[int] = []

    def fake_post(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        raise httpx.ConnectError("network down")

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)

    with pytest.raises(LLMRequestError, match="network down"):
        extract_scene_intent("make 2 cubes", {})

    assert len(calls) == 1


def test_extract_scene_intent_logs_timeout_and_completion_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)
    events: list[str] = []

    responses = [
        httpx.ReadTimeout("timed out"),
        FakeLLMResponse(
            '{"status":"ok","confidence":0.9,"reasoning":"test","entities":[{"logical_id":"plane_left","kind":"plane","quantity":1}]}'
        ),
    ]

    def fake_post(*args, **kwargs):
        del args, kwargs
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)
    monkeypatch.setattr(
        llm_client_module,
        "log_structured",
        lambda logger, event, payload, level="info": events.append(event),
    )

    extract_scene_intent("make a plane", {})

    assert "llm_request_started" in events
    assert "llm_request_timed_out" in events
    assert "llm_request_completed" in events


def test_extract_scene_intent_timeout_surfaces_structured_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)

    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ReadTimeout("timed out")),
    )

    with pytest.raises(LLMTimeoutError, match="timed out"):
        extract_scene_intent("make 2 cubes", {})


def test_plan_returns_safe_failure_when_pipeline_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "build_scene_pipeline",
        lambda prompt, scene_state, max_construction_steps=None: ScenePipelineResult(
            status="error",
            message="scene intent parse failed",
            prompt=prompt,
            error="scene intent parse failed",
        ),
    )

    result = plan("create a cube", {})

    assert result == PlannerResult(
        status="error",
        actions=[],
        message="No actions returned: scene intent parse failed",
    )


def test_plan_accepts_valid_compiled_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled_plan = CompiledConstructionPlan(
        actions=[
            {
                "action_id": "create_cube_1",
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
        continue_loop=False,
        expected_outcome="Scene intent is fully satisfied.",
    )

    monkeypatch.setattr(
        planner_module,
        "build_scene_pipeline",
        lambda prompt, scene_state, max_construction_steps=None: ScenePipelineResult(
            status="ok",
            message="planned",
            prompt=prompt,
            scene_intent=SceneIntent(
                status="ok",
                confidence=0.9,
                reasoning="test",
                entities=[SceneEntity(logical_id="cube_pair", kind="cube", quantity=1)],
            ),
            compiled_plan=compiled_plan,
        ),
    )

    result = plan("create a cube", {})

    assert result.status == "ok"
    assert result.actions == compiled_plan.actions
