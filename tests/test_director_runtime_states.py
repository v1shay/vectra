from __future__ import annotations

import httpx
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


def _endpoint() -> EndpointConfig:
    return EndpointConfig(
        provider="openai-director",
        family="openai",
        base_url="https://api.openai.test/v1",
        api_key="test-openai",
        model="gpt-5.1",
        transport="responses",
    )


def _responses_request() -> ProviderRequest:
    return ProviderRequest(
        instructions="You are the director.",
        user_input="Create a cube.",
        tools=[
            {
                "type": "function",
                "name": "mesh.create_primitive",
                "description": "Create a primitive mesh.",
                "parameters": {"type": "object", "properties": {"type": {"type": "string"}}},
            }
        ],
    )


def _http_response(status_code: int, payload: object, *, text: str | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.test/v1/responses")
    if text is not None:
        return httpx.Response(status_code, text=text, request=request)
    return httpx.Response(status_code, json=payload, request=request)


def test_http_provider_retries_retryable_500_before_success(monkeypatch) -> None:
    responses = [
        _http_response(500, {"error": {"message": "temporary failure"}}),
        _http_response(
            200,
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "mesh.create_primitive",
                        "arguments": '{"type":"cube","name":"Cube"}',
                    }
                ]
            },
        ),
    ]
    calls: list[str] = []

    def fake_post(*args, **kwargs):
        del args, kwargs
        calls.append("post")
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OpenAIResponsesAdapter().invoke(
        _endpoint(),
        _responses_request(),
        timeout=3.0,
        max_retries=1,
    )

    assert len(calls) == 2
    assert result.parsed is not None
    assert result.parsed.tool_calls[0].name == "mesh.create_primitive"
    assert result.attempt.response_metadata["retry_count"] == 1
    assert [item["status_code"] for item in result.attempt.response_metadata["transport_attempts"]] == [500, 200]


def test_http_provider_fail_fast_for_non_retryable_400(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post(*args, **kwargs):
        del args, kwargs
        calls.append("post")
        return _http_response(400, {"error": {"message": "bad request"}})

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OpenAIResponsesAdapter().invoke(
        _endpoint(),
        _responses_request(),
        timeout=3.0,
        max_retries=2,
    )

    assert len(calls) == 1
    assert result.parsed is None
    assert result.attempt.runtime_state == "provider_transport_failure"
    assert result.attempt.response_metadata["status_code"] == 400
    assert result.attempt.response_metadata["retry_count"] == 0
    assert result.attempt.response_metadata["transport_attempts"][0]["retryable"] is False


def test_http_provider_retries_json_decode_failure(monkeypatch) -> None:
    responses = [
        _http_response(200, {}, text="not-json"),
        _http_response(
            200,
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "task.clarify",
                        "arguments": '{"question":"q","reason":"r"}',
                    }
                ]
            },
        ),
    ]
    calls: list[str] = []

    def fake_post(*args, **kwargs):
        del args, kwargs
        calls.append("post")
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OpenAIResponsesAdapter().invoke(
        _endpoint(),
        _responses_request(),
        timeout=3.0,
        max_retries=1,
    )

    assert len(calls) == 2
    assert result.parsed is not None
    assert result.parsed.tool_calls[0].name == "task.clarify"
    assert result.attempt.response_metadata["retry_count"] == 1
    assert result.attempt.response_metadata["transport_attempts"][0]["error_type"] == "JSONDecodeError"


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


def test_structured_responses_requests_apply_safe_output_caps() -> None:
    endpoint = EndpointConfig(
        provider="openai-director",
        family="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-openai",
        model="gpt-5.4-mini",
        transport="responses",
    )
    adapter = OpenAIResponsesAdapter()

    request_with_tools = adapter.build_request(
        endpoint,
        ProviderRequest(
            instructions="You are the director.",
            user_input="Create a cube.",
            tools=[
                {
                    "type": "function",
                    "name": "mesh.create_primitive",
                    "description": "Create a primitive mesh.",
                    "parameters": {"type": "object", "properties": {"type": {"type": "string"}}},
                }
            ],
        ),
    )
    request_without_tools = adapter.build_request(
        endpoint,
        ProviderRequest(
            instructions="You are the controller.",
            user_input="Create a cube.",
            tools=[],
        ),
    )

    assert request_with_tools.payload["max_output_tokens"] == 1024
    assert request_with_tools.request_metadata["max_output_tokens"] == 1024
    assert request_without_tools.payload["max_output_tokens"] == 512
    assert request_without_tools.request_metadata["max_output_tokens"] == 512


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


def test_broad_prompt_single_action_triggers_validation_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: ControllerDecision(),
    )
    responses = [
        ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will start with the floor.",
                tool_calls=[ToolCall(name="mesh.create_primitive", arguments={"type": "plane", "name": "Ground"})],
                response_type="tool_calls",
            ),
            runtime_state="valid_action_batch_ready",
        ),
        ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will establish the floor and the key light together.",
                tool_calls=[
                    ToolCall(name="mesh.create_primitive", arguments={"type": "plane", "name": "Ground"}),
                    ToolCall(name="light.create", arguments={"type": "AREA"}),
                ],
                response_type="tool_calls",
            ),
            runtime_state="valid_action_batch_ready",
        ),
    ]
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools, allow_complete=False: responses.pop(0),
    )

    turn = DirectorLoop().step(_context("make a coherent cinematic room"))

    assert turn.status == "ok"
    assert turn.metadata["validation_retry_used"] is True
    assert len(turn.metadata["actions"]) == 2


def test_invalid_tool_after_retry_returns_tool_validation_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_controller",
        lambda prompt, scene_state: ControllerDecision(),
    )
    responses = [
        ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will create a cylinder.",
                tool_calls=[ToolCall(name="mesh.create_cylinder", arguments={"name": "LampBase"})],
                response_type="tool_calls",
            ),
            runtime_state="valid_action_batch_ready",
        ),
        ProviderResult(
            provider="openai-director",
            model="gpt-5.1",
            parsed=ParsedProviderResponse(
                assistant_text="I will create a cylinder.",
                tool_calls=[ToolCall(name="mesh.create_cylinder", arguments={"name": "LampBase"})],
                response_type="tool_calls",
            ),
            runtime_state="valid_action_batch_ready",
        ),
    ]
    monkeypatch.setattr(
        "agent_runtime.director.loop.call_director",
        lambda prompt_text, tools, allow_complete=False: responses.pop(0),
    )

    turn = DirectorLoop().step(_context("make a nice lamp"))

    assert turn.status == "error"
    assert turn.metadata["runtime_state"] == "tool_validation_failure"
    assert turn.metadata["validation_retry_used"] is True
    assert "mesh.create_cylinder" in turn.metadata["failure_reason"]
