from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from vectra.utils.logging import get_vectra_logger, log_structured

from .config import EndpointConfig
from .models import (
    ParsedProviderResponse,
    ProviderAdapterCapabilities,
    ProviderAttempt,
    RuntimeState,
    ToolCall,
)

_LOGGER = get_vectra_logger("vectra.runtime.director.adapters")


class ProviderError(Exception):
    """Raised when a provider adapter is unavailable or misconfigured."""

    def __init__(
        self,
        message: str,
        *,
        runtime_state: RuntimeState = "provider_transport_failure",
        attempts: list[ProviderAttempt] | None = None,
    ) -> None:
        super().__init__(message)
        self.runtime_state = runtime_state
        self.attempts = attempts or []


class ProviderTimeoutError(ProviderError):
    """Raised when a provider adapter times out."""


@dataclass(frozen=True)
class ProviderRequest:
    instructions: str
    user_input: str | list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    corrective_hint: str | None = None


@dataclass(frozen=True)
class ProviderHttpRequest:
    path: str
    payload: dict[str, Any]
    request_metadata: dict[str, Any]


@dataclass(frozen=True)
class AdapterCallResult:
    parsed: ParsedProviderResponse | None
    attempt: ProviderAttempt
    capabilities: ProviderAdapterCapabilities


def _extract_json_object(text: str) -> tuple[dict[str, Any], bool]:
    normalized = text.strip()
    if not normalized:
        return {}, False
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(normalized[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, True
        except json.JSONDecodeError:
            return {}, False
    return {}, False


def _serialized_length(value: str | list[dict[str, Any]]) -> int:
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, sort_keys=True))


def _tool_prompt_text(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "No tools are available."
    lines: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        description = str(tool.get("description", "")).strip()
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        argument_names = sorted(
            key for key in properties.keys() if isinstance(key, str) and key.strip()
        )
        args_text = ", ".join(argument_names) if argument_names else "no arguments"
        lines.append(f"- {name}: {description} Args: {args_text}")
    return "\n".join(lines) if lines else "No tools are available."


def _parse_status_to_runtime_state(parse_status: str) -> RuntimeState:
    if parse_status == "tool_call_parse_failure":
        return "tool_call_parse_failure"
    if parse_status == "no_action_response":
        return "no_action_response"
    return "valid_action_batch_ready"


def _response_type_for_tool_name(tool_name: str) -> str:
    if tool_name == "task.complete":
        return "complete"
    if tool_name == "task.clarify":
        return "clarify"
    return "tool_calls"


def _responses_max_output_tokens(tool_count: int) -> int:
    # Keep OpenRouter-compatible Responses requests inside a sane credit budget.
    return 1024 if tool_count else 512


class BaseProviderAdapter(ABC):
    family: str = ""
    transport: str = ""
    capabilities = ProviderAdapterCapabilities()

    @abstractmethod
    def build_request(self, endpoint: EndpointConfig, request: ProviderRequest) -> ProviderHttpRequest:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        endpoint: EndpointConfig,
        http_request: ProviderHttpRequest,
        *,
        timeout: float,
        max_retries: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, raw_response: dict[str, Any]) -> ParsedProviderResponse:
        raise NotImplementedError

    def invoke(
        self,
        endpoint: EndpointConfig,
        request: ProviderRequest,
        *,
        timeout: float,
        max_retries: int,
    ) -> AdapterCallResult:
        http_request = self.build_request(endpoint, request)
        try:
            raw_response = self.execute(
                endpoint,
                http_request,
                timeout=timeout,
                max_retries=max_retries,
            )
        except ProviderError as exc:
            runtime_state = getattr(exc, "runtime_state", "provider_transport_failure")
            attempt = ProviderAttempt(
                provider=endpoint.provider,
                model=endpoint.model,
                transport=endpoint.transport,
                runtime_state=runtime_state,
                failure_reason=str(exc),
                request_metadata=http_request.request_metadata,
                response_metadata={"error_type": exc.__class__.__name__},
            )
            log_structured(
                _LOGGER,
                "provider_parse_result",
                {
                    "provider": endpoint.provider,
                    "model": endpoint.model,
                    "transport": endpoint.transport,
                    "runtime_state": runtime_state,
                    "failure_reason": str(exc),
                },
                level="error",
            )
            return AdapterCallResult(parsed=None, attempt=attempt, capabilities=self.capabilities)

        parsed = self.parse_response(raw_response)
        response_metadata = {
            "response_type": parsed.response_type,
            "parsed_tool_call_count": len(parsed.tool_calls),
            "raw_response_keys": sorted(raw_response.keys())[:20],
        }
        log_structured(
            _LOGGER,
            "provider_response_received",
            {
                "provider": endpoint.provider,
                "model": endpoint.model,
                "transport": endpoint.transport,
                "response_type": parsed.response_type,
                "raw_response_keys": response_metadata["raw_response_keys"],
            },
        )
        runtime_state = _parse_status_to_runtime_state(parsed.parse_status)
        attempt = ProviderAttempt(
            provider=endpoint.provider,
            model=endpoint.model,
            transport=endpoint.transport,
            runtime_state=runtime_state,
            response_type=parsed.response_type,
            parsed_tool_call_count=len(parsed.tool_calls),
            failure_reason=parsed.failure_reason,
            request_metadata=http_request.request_metadata,
            response_metadata=response_metadata,
        )
        log_structured(
            _LOGGER,
            "provider_parse_result",
            {
                "provider": endpoint.provider,
                "model": endpoint.model,
                "transport": endpoint.transport,
                "response_type": parsed.response_type,
                "parse_status": parsed.parse_status,
                "runtime_state": runtime_state,
                "parsed_tool_call_count": len(parsed.tool_calls),
                "failure_reason": parsed.failure_reason,
            },
            level="warning" if runtime_state != "valid_action_batch_ready" else "info",
        )
        return AdapterCallResult(parsed=parsed, attempt=attempt, capabilities=self.capabilities)


class HttpJsonProviderAdapter(BaseProviderAdapter):
    def _request_with_logging(
        self,
        endpoint: EndpointConfig,
        *,
        http_request: ProviderHttpRequest,
        timeout: float,
        max_retries: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            started = time.perf_counter()
            log_structured(
                _LOGGER,
                "llm_request_started",
                {
                    "provider": endpoint.provider,
                    "model": endpoint.model,
                    "transport": endpoint.transport,
                    "attempt": attempt,
                    **http_request.request_metadata,
                },
            )
            try:
                headers = {"Content-Type": "application/json"}
                if endpoint.api_key:
                    headers["Authorization"] = f"Bearer {endpoint.api_key}"
                response = httpx.post(
                    f"{endpoint.base_url}{http_request.path}",
                    headers=headers,
                    json=http_request.payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ProviderError("Provider returned a non-object JSON payload")
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_structured(
                    _LOGGER,
                    "llm_request_completed",
                    {
                        "provider": endpoint.provider,
                        "model": endpoint.model,
                        "transport": endpoint.transport,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                return payload
            except httpx.TimeoutException as exc:
                last_error = exc
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_structured(
                    _LOGGER,
                    "llm_request_timed_out",
                    {
                        "provider": endpoint.provider,
                        "model": endpoint.model,
                        "transport": endpoint.transport,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                    },
                    level="warning",
                )
                if attempt > max_retries:
                    raise ProviderTimeoutError(
                        str(exc) or "provider request timed out",
                        runtime_state="provider_deadline_exceeded",
                    ) from exc
            except (httpx.HTTPError, ValueError) as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                error_payload = ""
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    try:
                        error_payload = exc.response.text[:500]
                    except Exception:  # pragma: no cover - defensive logging only
                        error_payload = ""
                log_structured(
                    _LOGGER,
                    "provider_failure",
                    {
                        "provider": endpoint.provider,
                        "model": endpoint.model,
                        "transport": endpoint.transport,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                        "error": str(exc),
                        "error_payload": error_payload,
                    },
                    level="error",
                )
                raise ProviderError(str(exc) or "provider request failed") from exc
        raise ProviderTimeoutError(
            str(last_error) if last_error else "provider request timed out",
            runtime_state="provider_deadline_exceeded",
        )

    def execute(
        self,
        endpoint: EndpointConfig,
        http_request: ProviderHttpRequest,
        *,
        timeout: float,
        max_retries: int,
    ) -> dict[str, Any]:
        return self._request_with_logging(
            endpoint,
            http_request=http_request,
            timeout=timeout,
            max_retries=max_retries,
        )


class StructuredResponsesAdapter(HttpJsonProviderAdapter):
    capabilities = ProviderAdapterCapabilities(
        structured_tools=True,
        embeds_tools_in_prompt=False,
        supports_parallel_tool_calls=True,
    )

    def build_request(self, endpoint: EndpointConfig, request: ProviderRequest) -> ProviderHttpRequest:
        instructions = request.instructions
        if request.corrective_hint:
            instructions = f"{instructions}\n\n{request.corrective_hint}"
        max_output_tokens = _responses_max_output_tokens(len(request.tools))
        payload = {
            "model": endpoint.model,
            "input": request.user_input,
            "instructions": instructions,
            "tools": request.tools,
            "max_output_tokens": max_output_tokens,
        }
        if request.tools:
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        return ProviderHttpRequest(
            path="/responses",
            payload=payload,
            request_metadata={
                "path": "/responses",
                "tool_count": len(request.tools),
                "input_chars": _serialized_length(request.user_input),
                "instruction_chars": len(instructions),
                "structured_tools": True,
                "prompt_embeds_tools": False,
                "max_output_tokens": max_output_tokens,
            },
        )

    def parse_response(self, raw_response: dict[str, Any]) -> ParsedProviderResponse:
        assistant_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        invalid_tool_calls = 0
        for item in raw_response.get("output", []):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    content_type = content.get("type")
                    if content_type in {"output_text", "text"}:
                        text = content.get("text") or content.get("value") or ""
                        if isinstance(text, str) and text.strip():
                            assistant_parts.append(text.strip())
            elif item_type in {"function_call", "tool_call"}:
                name = item.get("name")
                arguments = item.get("arguments")
                if not isinstance(name, str) or not name.strip():
                    invalid_tool_calls += 1
                    continue
                if isinstance(arguments, str):
                    parsed_arguments, success = _extract_json_object(arguments)
                    if not success:
                        invalid_tool_calls += 1
                        continue
                elif isinstance(arguments, dict):
                    parsed_arguments = arguments
                else:
                    invalid_tool_calls += 1
                    continue
                tool_calls.append(ToolCall(name=name.strip(), arguments=parsed_arguments))
            elif item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    for content in summary:
                        if isinstance(content, dict):
                            text = content.get("text")
                            if isinstance(text, str) and text.strip():
                                assistant_parts.append(text.strip())

        output_text = raw_response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            assistant_parts.append(output_text.strip())

        assistant_text = "\n".join(part for part in assistant_parts if part).strip()
        if tool_calls:
            response_type = _response_type_for_tool_name(tool_calls[0].name)
            return ParsedProviderResponse(
                assistant_text=assistant_text,
                tool_calls=tool_calls,
                response_type=response_type,
                parse_status="ok",
                raw_response=raw_response,
            )
        if invalid_tool_calls:
            return ParsedProviderResponse(
                assistant_text=assistant_text,
                response_type="malformed_tool_calls",
                parse_status="tool_call_parse_failure",
                failure_reason="Provider returned malformed structured tool calls.",
                raw_response=raw_response,
            )
        response_type = "message_only" if assistant_text else "empty"
        return ParsedProviderResponse(
            assistant_text=assistant_text,
            response_type=response_type,
            parse_status="no_action_response",
            failure_reason="Provider returned no tool calls.",
            raw_response=raw_response,
        )


class OpenAIResponsesAdapter(StructuredResponsesAdapter):
    family = "openai"
    transport = "responses"


class XAIResponsesAdapter(StructuredResponsesAdapter):
    family = "xai"
    transport = "responses"


class OllamaJsonEnvelopeAdapter(HttpJsonProviderAdapter):
    family = "ollama"
    transport = "ollama_json_envelope"
    capabilities = ProviderAdapterCapabilities(
        structured_tools=False,
        embeds_tools_in_prompt=True,
        supports_parallel_tool_calls=True,
    )

    def build_request(self, endpoint: EndpointConfig, request: ProviderRequest) -> ProviderHttpRequest:
        prompt = (
            f"{request.instructions}\n\n"
            "You must return a JSON object.\n"
            "Allowed top-level keys: assistant_text, tool_calls, tool_call, complete, clarify.\n"
            "tool_calls must be an array of 1 to 4 objects shaped like "
            "{\"name\": string, \"arguments\": object}.\n"
            "Use task.complete only when the current scene already satisfies the request.\n"
            "Do not return narration without one of those actionable keys.\n\n"
            "Available tools:\n"
            f"{_tool_prompt_text(request.tools)}\n\n"
            "User context:\n"
            f"{request.user_input}"
        )
        if request.corrective_hint:
            prompt = f"{prompt}\n\nCorrection:\n{request.corrective_hint}"
        return ProviderHttpRequest(
            path="/api/generate",
            payload={"model": endpoint.model, "prompt": prompt, "stream": False},
            request_metadata={
                "path": "/api/generate",
                "tool_count": len(request.tools),
                "input_chars": _serialized_length(request.user_input),
                "instruction_chars": len(request.instructions),
                "structured_tools": False,
                "prompt_embeds_tools": True,
            },
        )

    def parse_response(self, raw_response: dict[str, Any]) -> ParsedProviderResponse:
        response_text = str(raw_response.get("response", "")).strip()
        envelope, success = _extract_json_object(response_text)
        if not response_text:
            return ParsedProviderResponse(
                response_type="empty",
                parse_status="no_action_response",
                failure_reason="Provider returned an empty response.",
                raw_response=raw_response,
            )
        if not success:
            return ParsedProviderResponse(
                assistant_text=response_text,
                response_type="text",
                parse_status="no_action_response",
                failure_reason="Provider returned narration without a JSON action envelope.",
                raw_response=raw_response,
            )

        assistant_text = str(envelope.get("assistant_text", "")).strip()
        tool_calls: list[ToolCall] = []
        invalid_tool_calls = 0
        tool_calls_value = envelope.get("tool_calls")
        tool_call = envelope.get("tool_call")
        complete = envelope.get("complete")
        clarify = envelope.get("clarify")
        if isinstance(tool_calls_value, list):
            for item in tool_calls_value:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    invalid_tool_calls += 1
                    continue
                arguments = item.get("arguments", {})
                if not isinstance(arguments, dict):
                    invalid_tool_calls += 1
                    continue
                tool_calls.append(ToolCall(name=item["name"], arguments=arguments))
        elif isinstance(tool_call, dict):
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            if isinstance(name, str) and isinstance(arguments, dict):
                tool_calls.append(ToolCall(name=name, arguments=arguments))
            else:
                invalid_tool_calls += 1
        elif isinstance(complete, dict):
            tool_calls.append(ToolCall(name="task.complete", arguments=complete))
        elif isinstance(clarify, dict):
            tool_calls.append(ToolCall(name="task.clarify", arguments=clarify))

        if tool_calls:
            return ParsedProviderResponse(
                assistant_text=assistant_text,
                tool_calls=tool_calls,
                response_type=_response_type_for_tool_name(tool_calls[0].name),
                parse_status="ok",
                raw_response={"response": response_text},
            )
        if invalid_tool_calls:
            return ParsedProviderResponse(
                assistant_text=assistant_text,
                response_type="malformed_tool_calls",
                parse_status="tool_call_parse_failure",
                failure_reason="Provider returned a malformed JSON tool envelope.",
                raw_response={"response": response_text},
            )
        return ParsedProviderResponse(
            assistant_text=assistant_text,
            response_type="json_without_actions",
            parse_status="no_action_response",
            failure_reason="Provider returned a JSON envelope without actionable tool calls.",
            raw_response={"response": response_text},
        )


_ADAPTER_REGISTRY: dict[tuple[str, str], BaseProviderAdapter] = {}


def register_provider_adapter(adapter: BaseProviderAdapter) -> None:
    _ADAPTER_REGISTRY[(adapter.family, adapter.transport)] = adapter


def get_provider_adapter(endpoint: EndpointConfig) -> BaseProviderAdapter:
    adapter = _ADAPTER_REGISTRY.get((endpoint.family, endpoint.transport))
    if adapter is None:
        raise ProviderError(
            f"No provider adapter is registered for family={endpoint.family!r} transport={endpoint.transport!r}"
        )
    return adapter


def reset_provider_adapters() -> None:
    _ADAPTER_REGISTRY.clear()


def register_default_provider_adapters() -> None:
    for adapter in (
        OpenAIResponsesAdapter(),
        XAIResponsesAdapter(),
        OllamaJsonEnvelopeAdapter(),
    ):
        _ADAPTER_REGISTRY.setdefault((adapter.family, adapter.transport), adapter)


register_default_provider_adapters()
