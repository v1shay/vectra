from __future__ import annotations

import httpx
import pytest

import agent_runtime.planner as planner_module
from agent_runtime.llm_client import LLMResponseError, generate_actions
from agent_runtime.planner import plan
from tests.action_fixtures import (
    CREATE_CUBE_ACTIONS,
    CREATE_TWO_CUBES_ACTIONS,
    MOVE_CUBE_ACTIONS,
)


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


class FakeJSONResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("GET", "http://testserver/api/tags"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, object]:
        return self._payload


def _configure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_LLM_BASE_URL", "http://testserver")
    monkeypatch.setenv("VECTRA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("VECTRA_LLM_MODEL", "test-model")


def test_generate_actions_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: FakeLLMResponse(
            "```json\n"
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]\n'
            "```"
        ),
    )

    actions = generate_actions("create a cube", {})

    assert actions == CREATE_CUBE_ACTIONS


def test_generate_actions_retries_once_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    responses = iter(
        [
            FakeLLMResponse("not-json"),
            FakeLLMResponse(
                '[{"action_id":"move_cube","tool":"object.transform","params":{"object_name":"Cube","location":[2,0,0]}}]'
            ),
        ]
    )
    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: next(responses),
    )

    actions = generate_actions("move cube", {"active_object": "Cube"})

    assert actions == MOVE_CUBE_ACTIONS


def test_generate_actions_falls_back_to_generic_secondary_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    monkeypatch.setenv("VECTRA_LLM_FALLBACK_BASE_URL", "http://fallback.local/v1")
    monkeypatch.setenv("VECTRA_LLM_FALLBACK_API_KEY", "fallback-key")
    monkeypatch.setenv("VECTRA_LLM_FALLBACK_MODEL", "kimi-k2")
    monkeypatch.setattr("agent_runtime.llm_client._read_ollama_config", lambda: None)

    requested_urls: list[str] = []

    def fake_post(url: str, *args, **kwargs):
        requested_urls.append(url)
        if "testserver" in url:
            raise httpx.ConnectError("primary unavailable")
        return FakeLLMResponse(
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]'
        )

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)

    actions = generate_actions("create a cube", {})

    assert actions == CREATE_CUBE_ACTIONS
    assert requested_urls == [
        "http://testserver/chat/completions",
        "http://fallback.local/v1/chat/completions",
    ]


def test_generate_actions_falls_back_to_ollama_when_primary_config_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_MODEL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("VECTRA_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.get",
        lambda *args, **kwargs: FakeJSONResponse(
            {"models": [{"name": "qwen2.5-coder:32b"}, {"name": "deepseek-coder-v2:16b"}]}
        ),
    )

    requested_urls: list[str] = []

    def fake_post(url: str, *args, **kwargs):
        requested_urls.append(url)
        return FakeLLMResponse(
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]'
        )

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)

    actions = generate_actions("create a cube", {})

    assert actions == CREATE_CUBE_ACTIONS
    assert requested_urls == ["http://127.0.0.1:11434/v1/chat/completions"]


def test_plan_returns_safe_failure_on_invalid_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: (_ for _ in ()).throw(
            LLMResponseError("LLM returned invalid JSON: Expecting value")
        ),
    )

    result = plan("create a cube", {})

    assert result.actions == []
    assert result.message == "No actions returned: LLM returned invalid JSON: Expecting value"


def test_plan_handles_blank_prompt_safely() -> None:
    result = plan("   ", {})

    assert result.actions == []
    assert result.message == "No actions returned: empty prompt"


def test_plan_rejects_unknown_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [{"tool": "missing.tool", "params": {}}],
    )

    result = plan("do something", {})

    assert result.actions == []
    assert result.message == "No actions returned: Unknown tool 'missing.tool'"


def test_plan_rejects_duplicate_action_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {"action_id": "dup", "tool": "mesh.create_primitive", "params": {"primitive_type": "cube"}},
            {"action_id": "dup", "tool": "mesh.create_primitive", "params": {"primitive_type": "cube"}},
        ],
    )

    result = plan("create cubes", {})

    assert result.actions == []
    assert result.message == "No actions returned: Duplicate action_id 'dup'"


def test_plan_rejects_string_ref_shorthand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "move_cube",
                "tool": "object.transform",
                "params": {
                    "object_name": "$ref(create_cube.object_name)",
                    "location": [2, 0, 0],
                },
            }
        ],
    )

    result = plan("move cube", {})

    assert result.actions == []
    assert result.message == (
        "No actions returned: Action 'object.transform' must encode refs as objects at "
        "params.object_name"
    )


def test_plan_preserves_prompt_sensitive_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate_actions(prompt: str, scene_state: dict[str, object]) -> list[dict[str, object]]:
        del scene_state
        if prompt == "create a cube":
            return CREATE_CUBE_ACTIONS
        return CREATE_TWO_CUBES_ACTIONS

    monkeypatch.setattr(planner_module, "generate_actions", fake_generate_actions)

    single = plan("create a cube", {})
    multi = plan("create two cubes", {})

    assert single.actions == CREATE_CUBE_ACTIONS
    assert multi.actions == CREATE_TWO_CUBES_ACTIONS
    assert single.actions != multi.actions
