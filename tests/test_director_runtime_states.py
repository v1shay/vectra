from __future__ import annotations

import pytest

from agent_runtime.director.adapters import (
    AdapterCallResult,
    OllamaJsonEnvelopeAdapter,
    OpenAIResponsesAdapter,
    ProviderError,
    ProviderRequest,
    register_default_provider_adapters,
    register_provider_adapter,
    reset_provider_adapters,
)
from agent_runtime.director.config import EndpointConfig
from agent_runtime.director.loop import DirectorLoop, _build_director_prompt
from agent_runtime.director.models import (
    BudgetState,
    ControllerDecision,
    ParsedProviderResponse,
    ProviderAdapterCapabilities,
    ProviderAttempt,
    ProviderResult,
    ToolCall,
)


class FakeAdapter:
    family = "openai"
    transport = "fake-test"
    capabilities = ProviderAdapterCapabilities()

    def invoke(self, endpoint, request, *, timeout, max_retries):
        del request, timeout, max_retries
        return AdapterCallResult(
            parsed=ParsedProviderResponse(
                assistant_text="Creating a broad first step.",
                tool_calls=[ToolCall(name="mesh.create_primitive", arguments={"type": "plane", "name": "Ground"})],
                response_type="tool_calls",
            ),
            attempt=ProviderAttempt(
                provider=endpoint.provider,
                model=endpoint.model,
                transport=endpoint.transport,
                runtime_state="valid_action_batch_ready",
                response_type="tool_calls",
            ),
            capabilities=self.capabilities,
        )


def _context(prompt: str = "make something cool"):
    return __import__("agent_runtime.director.models", fromlist=["DirectorContext"]).DirectorContext(
        user_prompt=prompt,
        scene_state={
            "active_object": None,
            "selected_objects": [],
            "active_camera": None,
            "current_frame": 1,
            "scene_centroid": [0.0, 0.0, 0.0],
            "scene_bounds": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            "groups": [],
            "lights": [],
            "objects": [],
        },
        screenshot=None,
        history=[],
        iteration=1,
        execution_mode="vectra-dev",
        memory_results=[],
    )


def test_broad_scene_prompt_returns_actionable_first_step_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: ControllerDecision(),
    )
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools, allow_complete=False: ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I'll establish the scene with a base and a light.",
                tool_calls=[
                    ToolCall(name="mesh.create_primitive", arguments={"type": "plane", "name": "Ground"}),
                    ToolCall(name="light.create", arguments={"type": "AREA"}),
                ],
                response_type="tool_calls",
            ),
            runtime_state="valid_action_batch_ready",
        ),
    )

    turn = DirectorLoop().step(_context())

    assert turn.status == "ok"
    assert turn.metadata["runtime_state"] == "valid_action_batch_ready"
    assert turn.metadata["actions"]
    assert turn.metadata["actions"][0]["tool"] == "mesh.create_primitive"


def test_no_tool_call_response_is_rejected_and_logged(monkeypatch) -> None:
    endpoint = EndpointConfig(
        provider="openai-director",
        family="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-openai",
        model="gpt-5.1",
        transport="responses",
    )
    log_events: list[str] = []

    class NoActionAdapter:
        capabilities = ProviderAdapterCapabilities()

        def invoke(self, config, request, *, timeout, max_retries):
            del config, request, timeout, max_retries
            return AdapterCallResult(
                parsed=ParsedProviderResponse(
                    assistant_text="Here is a plan for the scene.",
                    response_type="message_only",
                    parse_status="no_action_response",
                    failure_reason="Provider returned no tool calls.",
                ),
                attempt=ProviderAttempt(
                    provider=endpoint.provider,
                    model=endpoint.model,
                    transport=endpoint.transport,
                    runtime_state="no_action_response",
                    response_type="message_only",
                    failure_reason="Provider returned no tool calls.",
                ),
                capabilities=ProviderAdapterCapabilities(),
            )

    monkeypatch.setattr("agent_runtime.director.providers._director_candidates", lambda runtime: [endpoint])
    monkeypatch.setattr("agent_runtime.director.providers.get_provider_adapter", lambda config: NoActionAdapter())
    monkeypatch.setattr(
        "agent_runtime.director.providers.log_structured",
        lambda logger, event, payload, level="info": log_events.append(event),
    )

    from agent_runtime.director.providers import call_director

    with pytest.raises(ProviderError) as excinfo:
        call_director(prompt_text="make something cool", tools=[], allow_complete=False)

    assert excinfo.value.runtime_state == "no_action_response"
    assert "non_actionable_response_rejected" in log_events


def test_different_providers_can_be_swapped_without_breaking_tool_parsing() -> None:
    openai_parsed = OpenAIResponsesAdapter().parse_response(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "mesh.create_primitive",
                    "arguments": '{"type": "plane", "name": "Ground"}',
                }
            ]
        }
    )
    ollama_parsed = OllamaJsonEnvelopeAdapter().parse_response(
        {
            "response": (
                '{"assistant_text":"Creating the floor.",'
                '"tool_calls":[{"name":"mesh.create_primitive","arguments":{"type":"plane","name":"Ground"}}]}'
            )
        }
    )

    assert openai_parsed.parse_status == "ok"
    assert ollama_parsed.parse_status == "ok"
    assert openai_parsed.tool_calls[0].name == ollama_parsed.tool_calls[0].name
    assert openai_parsed.tool_calls[0].arguments == ollama_parsed.tool_calls[0].arguments


def test_structured_adapters_omit_tool_schema_from_prompt_but_text_adapters_embed_it() -> None:
    prompt_text = _build_director_prompt(
        _context(),
        ControllerDecision(),
        BudgetState(
            complexity="medium",
            turn_budget=12,
            turns_used=0,
            turns_remaining=12,
            completion_mode_active=False,
            core_task_started=False,
        ),
    )
    tools = [
        {
            "type": "function",
            "name": "mesh.create_primitive",
            "description": "Create a primitive mesh.",
            "parameters": {"type": "object", "properties": {"type": {"type": "string"}}},
        }
    ]

    assert "Available tools" not in prompt_text
    assert "mesh.create_primitive" not in prompt_text

    ollama_request = OllamaJsonEnvelopeAdapter().build_request(
        EndpointConfig(
            provider="ollama-primary",
            family="ollama",
            base_url="http://127.0.0.1:11434",
            api_key=None,
            model="qwen2.5-coder:32b",
            transport="ollama_json_envelope",
        ),
        ProviderRequest(
            instructions="You are the director.",
            user_input=prompt_text,
            tools=tools,
        ),
    )

    assert "mesh.create_primitive" in ollama_request.payload["prompt"]


def test_custom_provider_adapter_can_be_registered_without_loop_changes(monkeypatch) -> None:
    monkeypatch.setenv("VECTRA_DIRECTOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("VECTRA_DIRECTOR_API_KEY", "test-openai")
    monkeypatch.setenv("VECTRA_DIRECTOR_MODEL", "gpt-5.1")
    monkeypatch.setenv("VECTRA_DIRECTOR_TRANSPORT", "fake-test")
    reset_provider_adapters()
    register_provider_adapter(FakeAdapter())
    register_default_provider_adapters()

    from agent_runtime.director.providers import call_director

    try:
        result = call_director(prompt_text="make something cool", tools=[], allow_complete=False)
    finally:
        reset_provider_adapters()
        register_default_provider_adapters()

    assert result.provider == "openai-director"
    assert result.tool_calls[0].name == "mesh.create_primitive"
    assert result.runtime_state == "valid_action_batch_ready"
