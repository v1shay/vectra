from __future__ import annotations

import httpx
import pytest

import agent_runtime.llm_client as llm_client_module
import agent_runtime.planner as planner_module
from agent_runtime.intent import IntentEnvelope, IntentStep
from agent_runtime.llm_client import (
    LLMConfigurationError,
    LLMEndpointConfig,
    LLMResponseError,
    extract_intent,
)
from agent_runtime.planner import plan


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


def test_system_prompt_uses_schema_rules_without_phrase_examples() -> None:
    prompt = llm_client_module._system_prompt()

    assert "Return JSON object only." in prompt
    assert "Do not produce actions." in prompt
    assert "target_ref='previous_step'" in prompt
    assert "move this shit forward 2" not in prompt
    assert "move ts forward 2" not in prompt
    assert "make some shit" not in prompt


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


def test_extract_intent_rejects_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)
    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: FakeLLMResponse(
            "```json\n"
            '{"status":"ok","confidence":0.9,"reason":"","steps":[{"action":"create","primitive_type":"cube","confidence":0.9}]}\n'
            "```"
        ),
    )

    with pytest.raises(LLMResponseError, match="LLM returned invalid JSON"):
        extract_intent("create a cube", {})


def test_extract_intent_logs_provider_usage_and_parsed_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_primary_env(monkeypatch)
    logged: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "agent_runtime.llm_client.httpx.post",
        lambda *args, **kwargs: FakeLLMResponse(
            '{"status":"ok","confidence":0.9,"reason":"","steps":[{"action":"create","primitive_type":"cube","confidence":0.9}]}'
        ),
    )
    monkeypatch.setattr(
        llm_client_module,
        "log_structured",
        lambda logger, event, payload, level="info": logged.append((event, payload)),
    )

    intent = extract_intent("create a cube", {})

    assert intent == IntentEnvelope(
        status="ok",
        confidence=0.9,
        reason="",
        steps=[IntentStep(action="create", primitive_type="cube", confidence=0.9)],
    )
    assert any(event == "llm_provider_used" for event, _payload in logged)
    assert any(event == "llm_parsed_intent" for event, _payload in logged)


def test_extract_intent_falls_back_to_ollama_only_after_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_primary_env(monkeypatch)
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
        if "api.mistral.ai" in url:
            raise httpx.ConnectError("primary unavailable")
        return FakeLLMResponse(
            '{"status":"ok","confidence":0.82,"reason":"","steps":[{"action":"create","primitive_type":"cube","confidence":0.82}]}'
        )

    monkeypatch.setattr("agent_runtime.llm_client.httpx.post", fake_post)

    intent = extract_intent("create a cube", {})

    assert intent.status == "ok"
    assert requested_urls == [
        "https://api.mistral.ai/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions",
    ]


def test_extract_intent_requires_primary_configuration_even_if_ollama_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("VECTRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("VECTRA_LLM_MODEL", raising=False)
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

    with pytest.raises(LLMConfigurationError, match="Missing primary LLM configuration"):
        extract_intent("create a cube", {})


def test_plan_returns_safe_failure_on_invalid_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "extract_intent",
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


def test_plan_rejects_unknown_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "extract_intent",
        lambda prompt, scene_state: IntentEnvelope(
            status="ok",
            confidence=0.9,
            reason="",
            steps=[IntentStep(action="create", primitive_type="cube", confidence=0.9)],
        ),
    )
    monkeypatch.setattr(
        planner_module,
        "plan_actions",
        lambda intent, scene_state: [{"action_id": "x", "tool": "missing.tool", "params": {}}],
    )

    result = plan("create cube", {})

    assert result.status == "error"
    assert result.actions == []
    assert result.message == "No actions returned: Unknown tool 'missing.tool'"


def test_plan_accepts_valid_multi_step_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_module,
        "extract_intent",
        lambda prompt, scene_state: IntentEnvelope(
            status="ok",
            confidence=0.9,
            reason="",
            steps=[
                IntentStep(action="create", primitive_type="cube", confidence=0.9),
                IntentStep(
                    action="transform",
                    target_ref="previous_step",
                    direction="forward",
                    magnitude=2,
                    transform_kind="location",
                    confidence=0.88,
                ),
            ],
        ),
    )

    result = plan("create a cube and move it forward 2", {})

    assert result.status == "ok"
    assert result.message == "planned"
    assert result.actions == [
        {
            "action_id": "step_1_create_cube",
            "tool": "mesh.create_primitive",
            "params": {"primitive_type": "cube", "location": [0.0, 0.0, 0.0]},
        },
        {
            "action_id": "step_2_transform_previous",
            "tool": "object.transform",
            "params": {
                "object_name": {"$ref": "step_1_create_cube.object_name"},
                "location": [0.0, 2.0, 0.0],
            },
        },
    ]
