from __future__ import annotations

import pytest

from agent_runtime.director.adapters import AdapterCallResult
from agent_runtime.director.config import load_runtime_config
from agent_runtime.director.models import (
    ParsedProviderResponse,
    ProviderAdapterCapabilities,
    ProviderAttempt,
    ToolCall,
)
from agent_runtime.director.providers import ProviderError, call_controller, call_director
from agent_runtime.llm_client import read_runtime_settings
from agent_runtime.planner import PlannerResult, plan


class FakeAdapter:
    capabilities = ProviderAdapterCapabilities()

    def __init__(self, handler) -> None:
        self._handler = handler

    def invoke(self, config, request, *, timeout, max_retries):
        return self._handler(config, request, timeout, max_retries)


def test_runtime_settings_read_new_director_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_DIRECTOR_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("VECTRA_DIRECTOR_MAX_RETRIES", "4")
    monkeypatch.setenv("VECTRA_DIRECTOR_STEP_DEADLINE_SECONDS", "55")
    monkeypatch.setenv("VECTRA_DIRECTOR_PROVIDER_ATTEMPT_BUDGET", "3")
    monkeypatch.setenv("VECTRA_DIRECTOR_TRANSPORT", "responses")
    monkeypatch.setenv("VECTRA_DIRECTOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("VECTRA_DIRECTOR_API_KEY", "test-openai")
    monkeypatch.setenv("VECTRA_OLLAMA_MODEL_PRIMARY", "qwen2.5-coder:32b")
    monkeypatch.setenv("VECTRA_OLLAMA_MODEL_SECONDARY", "deepseek-coder-v2:16b")

    settings = read_runtime_settings()
    runtime_config = load_runtime_config()

    assert settings.director_timeout_seconds == 30.0
    assert settings.director_max_retries == 4
    assert settings.director_step_deadline_seconds == 55.0
    assert settings.director_provider_attempt_budget == 3
    assert settings.ollama_primary_model == "qwen2.5-coder:32b"
    assert settings.ollama_secondary_model == "deepseek-coder-v2:16b"
    assert runtime_config.director is not None
    assert runtime_config.director.transport == "responses"
    assert runtime_config.director_step_deadline_seconds == 55.0
    assert runtime_config.director_provider_attempt_budget == 3


def test_controller_uses_xai_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_CONTROLLER_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setenv("VECTRA_CONTROLLER_API_KEY", "test-xai")
    monkeypatch.setenv("VECTRA_CONTROLLER_MODEL", "grok-4.2-reasoning")
    monkeypatch.setattr(
        "agent_runtime.director.providers.get_provider_adapter",
        lambda config: FakeAdapter(
            lambda config, request, timeout, max_retries: AdapterCallResult(
                parsed=ParsedProviderResponse(
                    assistant_text='{"needs_scene_context":true,"needs_visual_feedback":true,"complexity":"medium"}',
                    response_type="message_only",
                    parse_status="no_action_response",
                    failure_reason="Provider returned no tool calls.",
                ),
                attempt=ProviderAttempt(
                    provider=config.provider,
                    model=config.model,
                    transport=config.transport,
                    runtime_state="valid_action_batch_ready",
                    response_type="message_only",
                    request_metadata={"tool_count": len(request.tools)},
                ),
                capabilities=ProviderAdapterCapabilities(),
            )
        ),
    )

    decision = call_controller("make a cool scene", {"objects": []})

    assert decision.needs_scene_context is True
    assert decision.needs_visual_feedback is True
    assert decision.provider == "xai-controller"


def test_director_falls_back_to_xai_when_openai_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_DIRECTOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("VECTRA_DIRECTOR_API_KEY", "test-openai")
    monkeypatch.setenv("VECTRA_DIRECTOR_MODEL", "gpt-5.1")
    monkeypatch.setenv("VECTRA_CONTROLLER_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setenv("VECTRA_CONTROLLER_API_KEY", "test-xai")
    monkeypatch.setenv("VECTRA_CONTROLLER_MODEL", "grok-4.2-reasoning")

    seen_providers: list[str] = []

    monkeypatch.setattr(
        "agent_runtime.director.providers.get_provider_adapter",
        lambda config: FakeAdapter(
            lambda config, request, timeout, max_retries: (
                seen_providers.append(config.provider),
                AdapterCallResult(
                    parsed=None if config.provider == "openai-director" else ParsedProviderResponse(
                        tool_calls=[ToolCall(name="task.clarify", arguments={"question": "q", "reason": "r"})],
                        response_type="clarify",
                    ),
                    attempt=ProviderAttempt(
                        provider=config.provider,
                        model=config.model,
                        transport=config.transport,
                        runtime_state="provider_transport_failure" if config.provider == "openai-director" else "valid_action_batch_ready",
                        response_type="transport_failure" if config.provider == "openai-director" else "clarify",
                        failure_reason="openai unavailable" if config.provider == "openai-director" else "",
                        request_metadata={"tool_count": len(request.tools)},
                    ),
                    capabilities=ProviderAdapterCapabilities(),
                ),
            )[-1]
        ),
    )

    result = call_director(prompt_text="prompt", tools=[], allow_complete=False)

    assert seen_providers == ["openai-director", "xai-director-fallback"]
    assert result.provider == "xai-director-fallback"
    assert result.runtime_state == "fallback_provider_invoked"


def test_director_falls_back_to_ollama_after_cloud_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTRA_DIRECTOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("VECTRA_DIRECTOR_API_KEY", "test-openai")
    monkeypatch.setenv("VECTRA_CONTROLLER_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setenv("VECTRA_CONTROLLER_API_KEY", "test-xai")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("VECTRA_OLLAMA_MODEL_PRIMARY", "qwen2.5-coder:32b")
    monkeypatch.setenv("VECTRA_OLLAMA_MODEL_SECONDARY", "deepseek-coder-v2:16b")

    monkeypatch.setattr(
        "agent_runtime.director.providers.get_provider_adapter",
        lambda config: FakeAdapter(
            lambda config, request, timeout, max_retries: AdapterCallResult(
                parsed=(
                    None
                    if config.family != "ollama"
                    else ParsedProviderResponse(
                        tool_calls=[ToolCall(name="task.clarify", arguments={"question": "q", "reason": "r"})],
                        response_type="clarify",
                    )
                ),
                attempt=ProviderAttempt(
                    provider=config.provider,
                    model=config.model,
                    transport=config.transport,
                    runtime_state="provider_transport_failure" if config.family != "ollama" else "valid_action_batch_ready",
                    response_type="transport_failure" if config.family != "ollama" else "clarify",
                    failure_reason="cloud failed" if config.family != "ollama" else "",
                    request_metadata={"tool_count": len(request.tools)},
                ),
                capabilities=ProviderAdapterCapabilities(
                    structured_tools=config.family != "ollama",
                    embeds_tools_in_prompt=config.family == "ollama",
                ),
            )
        ),
    )

    result = call_director(prompt_text="prompt", tools=[], allow_complete=False)

    assert result.provider == "ollama-primary"
    assert result.model == "qwen2.5-coder:32b"


def test_plan_uses_director_loop_compatibility_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_turn = __import__("agent_runtime.director.models", fromlist=["DirectorTurn"]).DirectorTurn(
        status="ok",
        message="Prepared the next step.",
        narration="Creating the first cube.",
        understanding="The task wrapper asked for the next executable step only.",
        plan=["use mesh.create_primitive as the next step"],
        intended_actions=["mesh.create_primitive({'type': 'cube'})"],
        expected_outcome="A cube exists.",
        continue_loop=True,
        assumptions=[],
        metadata={
            "actions": [
                {
                    "action_id": "director_1_create",
                    "tool": "mesh.create_primitive",
                    "params": {"type": "cube", "location": [0.0, 0.0, 0.0]},
                }
            ]
        },
    )
    monkeypatch.setattr("agent_runtime.planner._DIRECTOR_LOOP.step", lambda context: fake_turn)

    result = plan("create 6 cubes", {"objects": []})

    assert result == PlannerResult(
        status="ok",
        actions=fake_turn.metadata["actions"],
        message="Prepared the next step.",
        assumptions=[],
        metadata=fake_turn.metadata,
    )
