from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from vectra.utils.logging import get_vectra_logger, log_structured

from .adapters import (
    ProviderError,
    ProviderRequest,
    ProviderTimeoutError,
    get_provider_adapter,
    register_default_provider_adapters,
    register_provider_adapter,
    reset_provider_adapters,
)
from .config import EndpointConfig, load_runtime_config
from .models import ControllerDecision, ProviderAttempt, ProviderResult, RuntimeState
from .prompts import controller_system_prompt, director_system_prompt

_LOGGER = get_vectra_logger("vectra.runtime.director.providers")


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


def build_controller_input(prompt: str, scene_state: dict[str, Any]) -> str:
    object_count = len(scene_state.get("objects", [])) if isinstance(scene_state.get("objects"), list) else 0
    return (
        f"Prompt: {prompt.strip()}\n"
        f"Scene object count: {object_count}\n"
        f"Active object: {scene_state.get('active_object')}\n"
        f"Selected objects: {scene_state.get('selected_objects', [])}"
    )


def build_director_input(prompt_text: str) -> str:
    return prompt_text


def _actionability_hint(*, allow_complete: bool) -> str:
    complete_rule = (
        "Use task.complete only when the current scene already satisfies the request."
        if allow_complete
        else "Do not use task.complete yet. Return the next actionable step instead."
    )
    return (
        "Your previous response was not actionable.\n"
        "Return exactly one of the following:\n"
        "1. a tool batch with 1 to 4 tool calls\n"
        "2. task.clarify with a short question and reason only if execution is truly impossible or unsafe\n"
        f"3. task.complete only if the current scene is already complete\n"
        f"{complete_rule}\n"
        "Do not return narration without actionable output."
    )


def _classify_director_response(
    *,
    tool_calls: list[Any],
    parse_status: str,
    failure_reason: str,
    allow_complete: bool,
) -> tuple[RuntimeState, str]:
    if parse_status == "tool_call_parse_failure":
        return "tool_call_parse_failure", failure_reason or "Provider returned malformed tool calls."
    if parse_status == "no_action_response":
        return "no_action_response", failure_reason or "Provider returned no actionable tool calls."
    if not tool_calls:
        return "no_action_response", "Provider returned no actionable tool calls."
    first_tool_name = getattr(tool_calls[0], "name", "")
    if first_tool_name == "task.complete" and not allow_complete:
        return "no_action_response", "Provider completed before returning a first actionable step."
    return "valid_action_batch_ready", ""


def _finalize_attempt(
    attempt: ProviderAttempt,
    *,
    runtime_state: RuntimeState,
    failure_reason: str = "",
) -> ProviderAttempt:
    response_metadata = dict(attempt.response_metadata)
    response_metadata["failure_reason"] = failure_reason
    return replace(
        attempt,
        runtime_state=runtime_state,
        failure_reason=failure_reason,
        response_metadata=response_metadata,
    )


def _attempts_to_dicts(attempts: list[ProviderAttempt]) -> list[dict[str, Any]]:
    return [
        {
            "provider": attempt.provider,
            "model": attempt.model,
            "transport": attempt.transport,
            "runtime_state": attempt.runtime_state,
            "response_type": attempt.response_type,
            "parsed_tool_call_count": attempt.parsed_tool_call_count,
            "failure_reason": attempt.failure_reason,
            "request_metadata": attempt.request_metadata,
            "response_metadata": attempt.response_metadata,
        }
        for attempt in attempts
    ]


def _aggregate_failure(attempts: list[ProviderAttempt]) -> tuple[RuntimeState, str]:
    if not attempts:
        return "provider_transport_failure", "No configured provider could satisfy the request."
    last_failure_reason = next(
        (attempt.failure_reason for attempt in reversed(attempts) if attempt.failure_reason),
        "No configured provider could satisfy the request.",
    )
    last_non_success = next(
        (
            attempt.runtime_state
            for attempt in reversed(attempts)
            if attempt.runtime_state != "valid_action_batch_ready"
        ),
        "provider_transport_failure",
    )
    return last_non_success, last_failure_reason


def _director_candidates(runtime) -> list[EndpointConfig]:
    candidates: list[EndpointConfig] = []
    if runtime.director is not None:
        candidates.append(runtime.director)
    if runtime.controller is not None:
        candidates.append(
            EndpointConfig(
                provider="xai-director-fallback",
                family="xai",
                base_url=runtime.controller.base_url,
                api_key=runtime.controller.api_key,
                model=runtime.controller.model,
                transport=runtime.controller.transport,
            )
        )
    candidates.append(
        EndpointConfig(
            provider="ollama-primary",
            family="ollama",
            base_url=runtime.ollama_host,
            api_key=None,
            model=runtime.ollama_primary_model,
            transport="ollama_json_envelope",
        )
    )
    if runtime.ollama_secondary_model != runtime.ollama_primary_model:
        candidates.append(
            EndpointConfig(
                provider="ollama-secondary",
                family="ollama",
                base_url=runtime.ollama_host,
                api_key=None,
                model=runtime.ollama_secondary_model,
                transport="ollama_json_envelope",
            )
        )
    return candidates


def call_controller(prompt: str, scene_state: dict[str, Any]) -> ControllerDecision:
    runtime = load_runtime_config()
    config = runtime.controller
    if config is None:
        return ControllerDecision()

    register_default_provider_adapters()
    try:
        adapter = get_provider_adapter(config)
    except ProviderError:
        return ControllerDecision()
    request = ProviderRequest(
        instructions=controller_system_prompt(),
        user_input=build_controller_input(prompt, scene_state),
        tools=[],
    )
    result = adapter.invoke(
        config,
        request,
        timeout=runtime.controller_timeout_seconds,
        max_retries=runtime.controller_max_retries,
    )
    if result.parsed is None or result.parsed.parse_status != "ok":
        return ControllerDecision()

    parsed = _extract_json_object(result.parsed.assistant_text)
    complexity = parsed.get("complexity")
    if complexity not in {"low", "medium", "high"}:
        complexity = "medium"
    return ControllerDecision(
        needs_scene_context=bool(parsed.get("needs_scene_context", True)),
        needs_visual_feedback=bool(parsed.get("needs_visual_feedback", False)),
        complexity=complexity,
        provider=config.provider,
        model=config.model,
        raw=parsed,
    )


def call_director(
    *,
    prompt_text: str,
    tools: list[dict[str, Any]],
    allow_complete: bool = False,
) -> ProviderResult:
    runtime = load_runtime_config()
    attempts: list[ProviderAttempt] = []
    register_default_provider_adapters()

    for candidate_index, config in enumerate(_director_candidates(runtime)):
        try:
            adapter = get_provider_adapter(config)
        except ProviderError as exc:
            attempts.append(
                ProviderAttempt(
                    provider=config.provider,
                    model=config.model,
                    transport=config.transport,
                    runtime_state="provider_transport_failure",
                    failure_reason=str(exc),
                )
            )
            continue
        if candidate_index > 0:
            log_structured(
                _LOGGER,
                "fallback_provider_invoked",
                {
                    "provider": config.provider,
                    "model": config.model,
                    "transport": config.transport,
                    "fallback_index": candidate_index,
                },
            )

        for corrective_retry in range(2):
            request = ProviderRequest(
                instructions=director_system_prompt(),
                user_input=build_director_input(prompt_text),
                tools=tools,
                corrective_hint=_actionability_hint(allow_complete=allow_complete) if corrective_retry else None,
            )
            call_result = adapter.invoke(
                config,
                request,
                timeout=runtime.director_timeout_seconds,
                max_retries=runtime.director_max_retries,
            )
            attempt = call_result.attempt
            parsed = call_result.parsed
            if parsed is None:
                attempts.append(attempt)
                break

            runtime_state, failure_reason = _classify_director_response(
                tool_calls=parsed.tool_calls,
                parse_status=parsed.parse_status,
                failure_reason=parsed.failure_reason,
                allow_complete=allow_complete,
            )
            attempts.append(
                _finalize_attempt(
                    attempt,
                    runtime_state=runtime_state,
                    failure_reason=failure_reason,
                )
            )
            if runtime_state == "valid_action_batch_ready":
                final_runtime_state: RuntimeState = (
                    "fallback_provider_invoked" if candidate_index > 0 else "valid_action_batch_ready"
                )
                if final_runtime_state != runtime_state:
                    attempts[-1] = _finalize_attempt(
                        attempts[-1],
                        runtime_state=final_runtime_state,
                        failure_reason="",
                    )
                return ProviderResult(
                    provider=config.provider,
                    model=config.model,
                    transport=config.transport,
                    parsed=parsed,
                    provider_chain=[f"{attempt.provider}:{attempt.model}" for attempt in attempts],
                    attempts=attempts,
                    adapter_capabilities=call_result.capabilities,
                    runtime_state=final_runtime_state,
                )

            log_structured(
                _LOGGER,
                "non_actionable_response_rejected",
                {
                    "provider": config.provider,
                    "model": config.model,
                    "transport": config.transport,
                    "runtime_state": runtime_state,
                    "failure_reason": failure_reason,
                    "corrective_retry": bool(corrective_retry),
                },
                level="warning",
            )
            if corrective_retry == 0:
                continue
            break

    runtime_state, failure_reason = _aggregate_failure(attempts)
    raise ProviderError(
        failure_reason,
        runtime_state=runtime_state,
        attempts=attempts,
    )


__all__ = [
    "ProviderError",
    "ProviderTimeoutError",
    "call_controller",
    "call_director",
    "get_provider_adapter",
    "register_default_provider_adapters",
    "register_provider_adapter",
    "reset_provider_adapters",
]
