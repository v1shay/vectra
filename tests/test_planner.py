from __future__ import annotations

import httpx
import pytest

import agent_runtime.llm_client as llm_client_module
import agent_runtime.planner as planner_module
from agent_runtime.llm_client import (
    LLMConfigurationError,
    LLMEndpointConfig,
    LLMResponseError,
    generate_actions,
)
from agent_runtime.planner import plan
from tests.action_fixtures import CREATE_CUBE_ACTIONS, CREATE_TWO_CUBES_ACTIONS, MOVE_CUBE_ACTIONS


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
def _disable_ollama_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module, "_read_ollama_config", lambda: None)


def _configure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_LLM_BASE_URL", "http://testserver")
    monkeypatch.setenv("VECTRA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("VECTRA_LLM_MODEL", "test-model")


def test_system_prompt_includes_coordinate_and_schema_guidance() -> None:
    prompt = llm_client_module._system_prompt()

    assert "+X = right" in prompt
    assert "+Y = forward" in prompt
    assert "-Y = backward" in prompt
    assert "Return JSON array only." in prompt
    assert "No extra top-level keys are allowed." in prompt
    assert "Vector params must be explicit JSON arrays in [x, y, z] order." in prompt
    assert "If the user says rotate without an axis, use the Z axis." in prompt
    assert "this shit" in prompt
    assert "shorthand like 'ts'" in prompt
    assert "If the user does not specify a concrete operation or target, return []" in prompt
    assert "Never fabricate coordinates for vague language like 'somewhere weird'." in prompt
    assert "create a cube and move it forward 2" in prompt


def test_user_content_emphasizes_scene_focus() -> None:
    content = llm_client_module._user_content(
        "slide it over",
        {
            "active_object": "Cube",
            "selected_objects": ["Cube"],
            "current_frame": 12,
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
    )

    assert "User request:\nslide it over" in content
    assert "active_object: Cube" in content
    assert "Scene objects summary:" in content
    assert "Cube | type=MESH | active=True | selected=True" in content
    assert "Raw scene_state JSON:" in content


def test_generate_actions_rejects_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: FakeLLMResponse(
            "```json\n"
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","location":[0,0,0]}}]\n'
            "```"
        ),
    )

    with pytest.raises(LLMResponseError, match="LLM returned invalid JSON"):
        generate_actions("create a cube", {})


def test_generate_actions_does_not_attempt_json_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    request_messages: list[list[dict[str, str]]] = []
    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: (
            request_messages.append(kwargs["json"]["messages"]) or FakeLLMResponse("not-json")
        ),
    )

    with pytest.raises(LLMResponseError, match="LLM returned invalid JSON"):
        generate_actions("move cube", {"active_object": "Cube"})

    assert len(request_messages) == 1


def test_generate_actions_logs_raw_output_and_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    logged: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: FakeLLMResponse(
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]'
        ),
    )
    monkeypatch.setattr(
        llm_client_module,
        "log_structured",
        lambda logger, event, payload, level="info": logged.append((event, payload)),
    )

    actions = generate_actions("create a cube", {})

    assert actions == CREATE_CUBE_ACTIONS
    assert ("llm_raw_output", {"provider": "primary", "content": '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]'}) in logged
    assert any(event == "llm_parsed_json" for event, payload in logged)


def test_generate_actions_falls_back_to_generic_secondary_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    monkeypatch.setenv("VECTRA_LLM_FALLBACK_BASE_URL", "http://fallback.local/v1")
    monkeypatch.setenv("VECTRA_LLM_FALLBACK_API_KEY", "fallback-key")
    monkeypatch.setenv("VECTRA_LLM_FALLBACK_MODEL", "kimi-k2")

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


def test_generate_actions_falls_back_to_local_ollama_when_primary_request_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    monkeypatch.setattr(
        llm_client_module,
        "_read_ollama_config",
        lambda: LLMEndpointConfig(
            name="ollama",
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            model="qwen2.5-coder:32b",
        ),
    )

    requested_urls: list[str] = []

    def fake_post(url: str, *args, **kwargs):
        requested_urls.append(url)
        if "testserver" in url:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("POST", url),
                response=httpx.Response(403),
            )
        return FakeLLMResponse(
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]'
        )

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)

    actions = generate_actions("create a cube", {})

    assert actions == CREATE_CUBE_ACTIONS
    assert requested_urls == [
        "http://testserver/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions",
    ]


def test_generate_actions_uses_local_ollama_when_primary_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_MODEL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(
        llm_client_module,
        "_read_ollama_config",
        lambda: LLMEndpointConfig(
            name="ollama",
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            model="qwen2.5-coder:32b",
        ),
    )
    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: FakeLLMResponse(
            '[{"action_id":"create_cube","tool":"mesh.create_primitive","params":{"primitive_type":"cube","name":"VectraCube","location":[0,0,0]}}]'
        ),
    )

    actions = generate_actions("create a cube", {})

    assert actions == CREATE_CUBE_ACTIONS


def test_generate_actions_requires_some_usable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECTRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_MODEL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(llm_client_module, "_read_ollama_config", lambda: None)

    with pytest.raises(LLMConfigurationError, match="Missing LLM configuration"):
        generate_actions("create a cube", {})


def test_plan_returns_safe_failure_on_invalid_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: (_ for _ in ()).throw(
            LLMResponseError("LLM returned invalid JSON: Expecting value")
        ),
    )

    result = plan("create a cube", {})

    assert result.status == "error"
    assert result.actions == []
    assert result.message == "No actions returned: LLM returned invalid JSON: Expecting value"


def test_plan_handles_blank_prompt_safely() -> None:
    result = plan("   ", {})

    assert result.status == "error"
    assert result.actions == []
    assert result.message == "No actions returned: empty prompt"


def test_plan_rejects_empty_action_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner_module, "generate_actions", lambda prompt, scene_state: [])

    result = plan("move cube somewhere weird", {})

    assert result.status == "error"
    assert result.actions == []
    assert result.message == "No actions returned: planner returned an empty action list"


def test_plan_rejects_unknown_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [{"tool": "missing.tool", "params": {}}],
    )

    result = plan("do something", {})

    assert result.status == "error"
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

    assert result.status == "error"
    assert result.actions == []
    assert result.message == "No actions returned: Duplicate action_id 'dup'"


def test_plan_rejects_extra_top_level_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "create_cube",
                "tool": "mesh.create_primitive",
                "params": {"primitive_type": "cube"},
                "note": "extra",
            }
        ],
    )

    result = plan("create cube", {})

    assert result.status == "error"
    assert result.message == "No actions returned: Action at index 0 has unknown keys: ['note']"


def test_plan_rejects_unknown_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "create_cube",
                "tool": "mesh.create_primitive",
                "params": {"primitive_type": "cube", "bogus": 1},
            }
        ],
    )

    result = plan("create cube", {})

    assert result.status == "error"
    assert result.message == "No actions returned: Action 'mesh.create_primitive' has unknown params: ['bogus']"


def test_plan_rejects_missing_required_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "move_cube",
                "tool": "object.transform",
                "params": {"location": [1, 2, 3]},
            }
        ],
    )

    result = plan("move cube", {})

    assert result.status == "error"
    assert result.message == "No actions returned: Action 'object.transform' is missing required params: ['object_name']"


def test_plan_rejects_non_array_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "move_cube",
                "tool": "object.transform",
                "params": {"object_name": "Cube", "location": {"x": 10}},
            }
        ],
    )

    result = plan("move cube to x 10", {})

    assert result.status == "error"
    assert result.message == (
        "No actions returned: Action 'object.transform' has an invalid $ref at location"
    )


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


def test_plan_rejects_unknown_ref_output_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "create_cube",
                "tool": "mesh.create_primitive",
                "params": {"primitive_type": "cube"},
            },
            {
                "action_id": "move_cube",
                "tool": "object.transform",
                "params": {
                    "object_name": {"$ref": "create_cube.missing"},
                    "location": [0, 2, 0],
                },
            },
        ],
    )

    result = plan("create and move cube", {})

    assert result.status == "error"
    assert result.message == (
        "No actions returned: Action 'object.transform' references unknown output 'missing' at object_name"
    )


def test_plan_accepts_valid_multi_step_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "generate_actions",
        lambda prompt, scene_state: [
            {
                "action_id": "create_cube",
                "tool": "mesh.create_primitive",
                "params": {"primitive_type": "cube", "location": [0, 0, 0]},
            },
            {
                "action_id": "move_cube",
                "tool": "object.transform",
                "params": {
                    "object_name": {"$ref": "create_cube.object_name"},
                    "location": [0, 2, 0],
                },
            },
        ],
    )

    result = plan("create a cube and move it forward 2", {})

    assert result.status == "ok"
    assert result.message == "planned"
    assert result.actions == [
        {
            "action_id": "create_cube",
            "tool": "mesh.create_primitive",
            "params": {"primitive_type": "cube", "location": [0, 0, 0]},
        },
        {
            "action_id": "move_cube",
            "tool": "object.transform",
            "params": {
                "object_name": {"$ref": "create_cube.object_name"},
                "location": [0, 2, 0],
            },
        },
    ]


def test_plan_preserves_prompt_sensitive_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate_actions(prompt: str, scene_state: dict[str, object]) -> list[dict[str, object]]:
        del scene_state
        if prompt == "create a cube":
            return CREATE_CUBE_ACTIONS
        return CREATE_TWO_CUBES_ACTIONS

    monkeypatch.setattr(planner_module, "generate_actions", fake_generate_actions)

    single = plan("create a cube", {})
    multi = plan("drop a second cube over there", {})

    assert single.status == "ok"
    assert multi.status == "ok"
    assert single.actions == CREATE_CUBE_ACTIONS
    assert multi.actions == CREATE_TWO_CUBES_ACTIONS
    assert single.actions != multi.actions


def test_plan_accepts_valid_simple_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner_module, "generate_actions", lambda prompt, scene_state: MOVE_CUBE_ACTIONS)

    result = plan("move cube", {})

    assert result.status == "ok"
    assert result.actions == MOVE_CUBE_ACTIONS
