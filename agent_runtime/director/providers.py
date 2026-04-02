from __future__ import annotations

import json
import time
from typing import Any

import httpx

from vectra.utils.logging import get_vectra_logger, log_structured

from .config import EndpointConfig, load_runtime_config
from .models import ControllerDecision, ProviderResult, ToolCall
from .prompts import controller_system_prompt, director_system_prompt

_LOGGER = get_vectra_logger("vectra.runtime.director.providers")


class ProviderError(Exception):
    """Raised when a provider cannot fulfill a request."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider times out."""


def _extract_json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        return {}
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(normalized[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _request_with_logging(
    config: EndpointConfig,
    *,
    path: str,
    payload: dict[str, Any],
    timeout: float,
    attempt: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    log_structured(
        _LOGGER,
        "llm_request_started",
        {
            "provider": config.provider,
            "model": config.model,
            "attempt": attempt,
        },
    )
    try:
        response = httpx.post(
            f"{config.base_url}{path}",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_structured(
            _LOGGER,
            "llm_request_timed_out",
            {
                "provider": config.provider,
                "model": config.model,
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
            },
            level="warning",
        )
        raise ProviderTimeoutError(str(exc) or "provider request timed out") from exc
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
                "provider": config.provider,
                "model": config.model,
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
                "error_payload": error_payload,
            },
            level="error",
        )
        raise ProviderError(str(exc) or "provider request failed") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log_structured(
        _LOGGER,
        "llm_request_completed",
        {
            "provider": config.provider,
            "model": config.model,
            "attempt": attempt,
            "elapsed_ms": elapsed_ms,
        },
    )
    return data if isinstance(data, dict) else {}


def _parse_responses_output(data: dict[str, Any]) -> ProviderResult:
    assistant_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in data.get("output", []):
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
                continue
            if isinstance(arguments, str):
                parsed_args = _extract_json_object(arguments)
            elif isinstance(arguments, dict):
                parsed_args = arguments
            else:
                parsed_args = {}
            tool_calls.append(ToolCall(name=name, arguments=parsed_args))
        elif item_type == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list):
                for content in summary:
                    if isinstance(content, dict):
                        text = content.get("text")
                        if isinstance(text, str) and text.strip():
                            assistant_parts.append(text.strip())
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        assistant_parts.append(output_text.strip())
    return ProviderResult(
        provider="",
        model="",
        assistant_text="\n".join(part for part in assistant_parts if part).strip(),
        tool_calls=tool_calls,
        raw_response=data,
    )


def _call_responses_api(
    config: EndpointConfig,
    *,
    instructions: str,
    user_input: str | list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
) -> ProviderResult:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            payload = {
                "model": config.model,
                "input": user_input,
                "instructions": instructions,
                "tools": tools,
            }
            if tools:
                payload["tool_choice"] = "auto"
                payload["parallel_tool_calls"] = True
            data = _request_with_logging(
                config,
                path="/responses",
                payload=payload,
                timeout=timeout,
                attempt=attempt,
            )
            result = _parse_responses_output(data)
            return ProviderResult(
                provider=config.provider,
                model=config.model,
                assistant_text=result.assistant_text,
                tool_calls=result.tool_calls,
                raw_response=data,
            )
        except ProviderTimeoutError as exc:
            last_error = exc
            if attempt > max_retries:
                break
        except ProviderError as exc:
            raise exc
    raise ProviderTimeoutError(str(last_error) if last_error else "provider request timed out")


def _call_ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    timeout: float,
    attempt: int,
) -> ProviderResult:
    started = time.perf_counter()
    log_structured(
        _LOGGER,
        "llm_request_started",
        {"provider": "ollama", "model": model, "attempt": attempt},
    )
    try:
        response = httpx.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_structured(
            _LOGGER,
            "llm_request_timed_out",
            {"provider": "ollama", "model": model, "attempt": attempt, "elapsed_ms": elapsed_ms},
            level="warning",
        )
        raise ProviderTimeoutError(str(exc) or "ollama request timed out") from exc
    except (httpx.HTTPError, ValueError) as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_structured(
            _LOGGER,
            "provider_failure",
            {
                "provider": "ollama",
                "model": model,
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            },
            level="error",
        )
        raise ProviderError(str(exc) or "ollama request failed") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log_structured(
        _LOGGER,
        "llm_request_completed",
        {"provider": "ollama", "model": model, "attempt": attempt, "elapsed_ms": elapsed_ms},
    )
    response_text = str(payload.get("response", "")).strip()
    parsed = _extract_json_object(response_text)
    tool_call = parsed.get("tool_call")
    tool_calls_value = parsed.get("tool_calls")
    complete = parsed.get("complete")
    clarify = parsed.get("clarify")
    tool_calls: list[ToolCall] = []
    if isinstance(tool_calls_value, list):
        for item in tool_calls_value:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(ToolCall(name=item["name"], arguments=arguments))
    elif isinstance(tool_call, dict) and isinstance(tool_call.get("name"), str):
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(ToolCall(name=tool_call["name"], arguments=arguments))
    elif isinstance(complete, dict):
        tool_calls.append(ToolCall(name="task.complete", arguments=complete))
    elif isinstance(clarify, dict):
        tool_calls.append(ToolCall(name="task.clarify", arguments=clarify))
    return ProviderResult(
        provider="ollama",
        model=model,
        assistant_text=str(parsed.get("assistant_text", "")).strip(),
        tool_calls=tool_calls,
        raw_response={"response": response_text},
        provider_chain=[f"ollama:{model}"],
    )


def build_controller_input(prompt: str, scene_state: dict[str, Any]) -> str:
    object_count = len(scene_state.get("objects", [])) if isinstance(scene_state.get("objects"), list) else 0
    return (
        f"Prompt: {prompt.strip()}\n"
        f"Scene object count: {object_count}\n"
        f"Active object: {scene_state.get('active_object')}\n"
        f"Selected objects: {scene_state.get('selected_objects', [])}"
    )


def call_controller(prompt: str, scene_state: dict[str, Any]) -> ControllerDecision:
    runtime = load_runtime_config()
    config = runtime.controller
    if config is None:
        return ControllerDecision()

    try:
        result = _call_responses_api(
            config,
            instructions=controller_system_prompt(),
            user_input=build_controller_input(prompt, scene_state),
            tools=[],
            timeout=runtime.controller_timeout_seconds,
            max_retries=runtime.controller_max_retries,
        )
    except ProviderError:
        return ControllerDecision()

    parsed = _extract_json_object(result.assistant_text)
    complexity = parsed.get("complexity")
    if complexity not in {"low", "medium", "high"}:
        complexity = "medium"
    return ControllerDecision(
        needs_scene_context=bool(parsed.get("needs_scene_context", True)),
        needs_visual_feedback=bool(parsed.get("needs_visual_feedback", False)),
        complexity=complexity,
        provider=result.provider,
        model=result.model,
        raw=parsed,
    )


def build_director_input(prompt_text: str) -> str:
    return prompt_text


def call_director(
    *,
    prompt_text: str,
    tools: list[dict[str, Any]],
) -> ProviderResult:
    runtime = load_runtime_config()
    last_error: Exception | None = None
    cloud_chain: list[EndpointConfig] = []
    if runtime.director is not None:
        cloud_chain.append(runtime.director)
    if runtime.controller is not None:
        cloud_chain.append(
            EndpointConfig(
                provider="xai-director-fallback",
                base_url=runtime.controller.base_url,
                api_key=runtime.controller.api_key,
                model=runtime.controller.model,
                transport=runtime.controller.transport,
            )
        )

    for config in cloud_chain:
        try:
            result = _call_responses_api(
                config,
                instructions=director_system_prompt(),
                user_input=build_director_input(prompt_text),
                tools=tools,
                timeout=runtime.director_timeout_seconds,
                max_retries=runtime.director_max_retries,
            )
            return ProviderResult(
                provider=result.provider,
                model=result.model,
                assistant_text=result.assistant_text,
                tool_calls=result.tool_calls,
                raw_response=result.raw_response,
                provider_chain=[f"{config.provider}:{config.model}"],
            )
        except ProviderError as exc:
            last_error = exc

    ollama_prompt = (
        f"{director_system_prompt()}\n\n"
        "Return a JSON object with optional keys assistant_text, tool_calls, tool_call, complete, or clarify.\n"
        "tool_calls must be an array of up to 4 items that each look like {\"name\": string, \"arguments\": object}.\n"
        "If you return a single tool call, you may use tool_call instead.\n"
        f"User context:\n{prompt_text}"
    )
    for attempt, model in enumerate(
        [runtime.ollama_primary_model, runtime.ollama_secondary_model],
        start=1,
    ):
        try:
            return _call_ollama_generate(
                host=runtime.ollama_host,
                model=model,
                prompt=ollama_prompt,
                timeout=runtime.director_timeout_seconds,
                attempt=attempt,
            )
        except ProviderError as exc:
            last_error = exc

    raise ProviderError(str(last_error) if last_error else "No configured provider could satisfy the request")
