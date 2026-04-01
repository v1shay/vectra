from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from vectra.utils.logging import get_vectra_logger, log_structured

try:
    from .scene_intent import SceneIntent, SceneIntentParseError, parse_scene_intent_content
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from scene_intent import SceneIntent, SceneIntentParseError, parse_scene_intent_content

DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_SCENE_OBJECT_LIMIT = 30
OLLAMA_DISCOVERY_TIMEOUT_SECONDS = 2.0
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_API_KEY = "ollama"
PREFERRED_OLLAMA_MODEL_HINTS = (
    "qwen2.5-coder",
    "deepseek-coder-v2",
    "qwen",
    "deepseek",
    "coder",
)


class LLMClientError(Exception):
    """Base exception for LLM planning client failures."""


class LLMConfigurationError(LLMClientError):
    """Raised when required LLM configuration is missing."""


class LLMRequestError(LLMClientError):
    """Raised when the LLM request cannot be completed."""


class LLMTimeoutError(LLMRequestError):
    """Raised when the LLM request timed out after all retries."""


class LLMResponseError(LLMClientError):
    """Raised when the LLM response is invalid."""


@dataclass(frozen=True)
class LLMEndpointConfig:
    name: str
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class LLMRuntimeSettings:
    timeout_seconds: float
    max_retries: int
    scene_object_limit: int


_LLM_LOGGER = get_vectra_logger("vectra.runtime.llm")


def _normalize_http_url(raw_url: str) -> str:
    normalized = raw_url.strip()
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


def _read_float_env(flag_name: str, default: float) -> float:
    raw_value = os.getenv(flag_name, "").strip()
    if not raw_value:
        return default
    try:
        return max(float(raw_value), 1.0)
    except ValueError:
        return default


def _read_int_env(flag_name: str, default: int) -> int:
    raw_value = os.getenv(flag_name, "").strip()
    if not raw_value:
        return default
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return default


def read_runtime_settings() -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        timeout_seconds=_read_float_env("VECTRA_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_retries=_read_int_env("VECTRA_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        scene_object_limit=_read_int_env("VECTRA_LLM_SCENE_OBJECT_LIMIT", DEFAULT_SCENE_OBJECT_LIMIT),
    )


def _read_env_config(
    *,
    name: str,
    base_url_var: str,
    api_key_var: str,
    model_var: str,
) -> LLMEndpointConfig | None:
    base_url = _normalize_http_url(os.getenv(base_url_var, ""))
    api_key = os.getenv(api_key_var, "").strip()
    model = os.getenv(model_var, "").strip()
    if not (base_url and api_key and model):
        return None
    return LLMEndpointConfig(name=name, base_url=base_url, api_key=api_key, model=model)


def _primary_config() -> LLMEndpointConfig | None:
    return _read_env_config(
        name="primary",
        base_url_var="VECTRA_LLM_BASE_URL",
        api_key_var="VECTRA_LLM_API_KEY",
        model_var="VECTRA_LLM_MODEL",
    )


def _ollama_root_url() -> str:
    raw_url = os.getenv("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_BASE_URL
    normalized = _normalize_http_url(raw_url)
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def _select_ollama_model(model_names: list[str]) -> str | None:
    if not model_names:
        return None

    lowered = [(model_name, model_name.lower()) for model_name in model_names]
    for hint in PREFERRED_OLLAMA_MODEL_HINTS:
        for model_name, lowered_name in lowered:
            if hint in lowered_name:
                return model_name
    return model_names[0]


def _discover_ollama_model(root_url: str) -> str | None:
    try:
        response = httpx.get(f"{root_url}/api/tags", timeout=OLLAMA_DISCOVERY_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list):
        return None

    model_names = [
        model.get("name")
        for model in raw_models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]
    return _select_ollama_model(model_names)


def _read_ollama_config() -> LLMEndpointConfig | None:
    model = os.getenv("VECTRA_OLLAMA_MODEL", "").strip()
    root_url = _ollama_root_url()
    if not model:
        model = _discover_ollama_model(root_url) or ""
    if not model:
        return None

    api_key = os.getenv("OLLAMA_API_KEY", "").strip() or DEFAULT_OLLAMA_API_KEY
    return LLMEndpointConfig(name="ollama", base_url=f"{root_url}/v1", api_key=api_key, model=model)


def _provider_chain() -> list[LLMEndpointConfig]:
    primary = _primary_config()
    if primary is None:
        raise LLMConfigurationError(
            "Missing primary LLM configuration. Set VECTRA_LLM_BASE_URL, "
            "VECTRA_LLM_API_KEY, and VECTRA_LLM_MODEL."
        )

    providers = [primary]
    ollama = _read_ollama_config()
    if ollama is not None:
        providers.append(ollama)
    return providers


def _scene_intent_schema_description() -> str:
    return json.dumps(
        {
            "status": "ok | no_action",
            "confidence": "number between 0 and 1",
            "reasoning": "string",
            "entities": [
                {
                    "logical_id": "stable logical id such as cube_pair or plane_left",
                    "kind": "cube | plane | uv_sphere",
                    "display_name": "optional display label",
                    "quantity": "integer >= 1",
                    "source": "create | reference | auto",
                    "reference_name": "optional exact scene object name when referencing",
                    "group_id": "optional logical group id",
                    "initial_transform": {
                        "offset": "[x, y, z] absolute placement intent from world origin",
                    },
                }
            ],
            "relationships": [
                {
                    "logical_id": "stable relationship id",
                    "relation_type": "left_of | right_of | above | below | next_to | relative_offset",
                    "source_id": "source logical entity id",
                    "target_id": "optional target logical entity id",
                    "target_group_id": "optional target group id",
                    "offset": "optional [x, y, z] for relative_offset",
                    "distance": "optional numeric spacing",
                    "metadata": {"touching": "optional boolean for stacking"},
                }
            ],
            "groups": [
                {
                    "logical_id": "group id",
                    "entity_ids": ["entity logical ids"],
                    "pronouns": ["them", "both", "all", "each"],
                }
            ],
            "construction_steps": [
                {
                    "logical_id": "stable step id",
                    "kind": "ensure_entity | apply_transform | resolve_group | satisfy_relation",
                    "entity_id": "optional entity id",
                    "relationship_id": "optional relationship id",
                    "group_id": "optional group id",
                    "offset": "optional [x, y, z]",
                }
            ],
            "uncertainty_notes": ["optional notes"],
            "metadata": {"pattern": "optional high-level pattern like staircase"},
        },
        indent=2,
        sort_keys=True,
    )


def _system_prompt() -> str:
    return (
        "You are Vectra's SceneIntent planner.\n"
        "Translate the user's request and compact scene summary into a typed scene construction plan.\n"
        "Return either:\n"
        "1) a JSON object matching the documented SceneIntent schema exactly, or\n"
        "2) a labeled text object with sections STATUS, CONFIDENCE, REASONING, ENTITIES, RELATIONSHIPS, "
        "GROUPS, CONSTRUCTION_STEPS, UNCERTAINTY, and METADATA.\n"
        "Do not return markdown. Do not emit tool calls. Do not emit Python. Do not describe Blender operators.\n"
        "Entities represent scene objects to create or reference, not executable actions.\n"
        "Relationships must be explicit and data-oriented.\n"
        "Prefer stable logical ids and explicit quantities.\n"
        "When a request contains quantity or plural references, add a group and use pronouns like them/both/all/each there.\n"
        "Use relation_type='relative_offset' for requests like move them apart or staircase layouts.\n"
        "Use status='no_action' when the request cannot be grounded safely.\n"
        "SceneIntent schema:\n"
        f"{_scene_intent_schema_description()}"
    )


def _scene_object_summary(scene_state: dict[str, Any], *, object_limit: int) -> str:
    objects = scene_state.get("objects", [])
    if not isinstance(objects, list) or not objects:
        return "- no scene objects"

    lines: list[str] = []
    for raw_object in objects[:object_limit]:
        if not isinstance(raw_object, dict):
            continue
        lines.append(
            "- "
            f"name={raw_object.get('name', '<unknown>')} "
            f"type={raw_object.get('type', '<unknown>')} "
            f"selected={raw_object.get('selected', False)} "
            f"active={raw_object.get('active', False)} "
            f"location={raw_object.get('location', [])} "
            f"rotation={raw_object.get('rotation_euler', [])} "
            f"scale={raw_object.get('scale', [])} "
            f"dimensions={raw_object.get('dimensions', [])}"
        )

    remaining = max(len(objects) - object_limit, 0)
    if remaining:
        lines.append(f"- truncated_additional_objects={remaining}")
    return "\n".join(lines) if lines else "- no scene objects"


def _user_content(prompt: str, scene_state: dict[str, Any], *, settings: LLMRuntimeSettings) -> str:
    return (
        f"User request:\n{prompt}\n\n"
        "Scene summary:\n"
        f"- active_object: {scene_state.get('active_object')}\n"
        f"- selected_objects: {scene_state.get('selected_objects', [])}\n"
        f"- object_count: {len(scene_state.get('objects', [])) if isinstance(scene_state.get('objects'), list) else 0}\n"
        "Compact object list:\n"
        f"{_scene_object_summary(scene_state, object_limit=settings.scene_object_limit)}"
    )


def _messages(prompt: str, scene_state: dict[str, Any], *, settings: LLMRuntimeSettings) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_content(prompt, scene_state, settings=settings)},
    ]


def _extract_message_content(response_json: dict[str, Any]) -> str:
    try:
        message_content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError("LLM response missing choices[0].message.content") from exc

    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        text_parts: list[str] = []
        for part in message_content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)

    raise LLMResponseError("LLM response content must be a string")


def _elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _log_started(config: LLMEndpointConfig, *, attempt: int, timeout_seconds: float) -> None:
    log_structured(
        _LLM_LOGGER,
        "llm_request_started",
        {
            "provider": config.name,
            "model": config.model,
            "attempt": attempt,
            "timeout_seconds": timeout_seconds,
        },
    )


def _log_completed(config: LLMEndpointConfig, *, attempt: int, elapsed_ms: int) -> None:
    log_structured(
        _LLM_LOGGER,
        "llm_request_completed",
        {
            "provider": config.name,
            "model": config.model,
            "attempt": attempt,
            "elapsed_ms": elapsed_ms,
        },
    )


def _log_timed_out(config: LLMEndpointConfig, *, attempt: int, elapsed_ms: int) -> None:
    log_structured(
        _LLM_LOGGER,
        "llm_request_timed_out",
        {
            "provider": config.name,
            "model": config.model,
            "attempt": attempt,
            "elapsed_ms": elapsed_ms,
        },
        level="warning",
    )


def _log_provider_failure(config: LLMEndpointConfig, *, attempt: int, elapsed_ms: int, message: str) -> None:
    log_structured(
        _LLM_LOGGER,
        "llm_provider_failure",
        {
            "provider": config.name,
            "model": config.model,
            "attempt": attempt,
            "elapsed_ms": elapsed_ms,
            "message": message,
        },
        level="warning",
    )


def _request_content_for_config(
    config: LLMEndpointConfig,
    messages: list[dict[str, str]],
    *,
    settings: LLMRuntimeSettings,
) -> str:
    attempts = settings.max_retries + 1
    last_error: LLMClientError | None = None

    for attempt in range(1, attempts + 1):
        started_at = time.perf_counter()
        _log_started(config, attempt=attempt, timeout_seconds=settings.timeout_seconds)
        try:
            response = httpx.post(
                f"{config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model,
                    "temperature": 0,
                    "messages": messages,
                },
                timeout=settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = _extract_message_content(payload)
            _log_completed(config, attempt=attempt, elapsed_ms=_elapsed_ms(started_at))
            log_structured(
                _LLM_LOGGER,
                "llm_raw_output",
                {"provider": config.name, "model": config.model, "attempt": attempt, "content": content},
            )
            return content
        except httpx.TimeoutException as exc:
            elapsed_ms = _elapsed_ms(started_at)
            _log_timed_out(config, attempt=attempt, elapsed_ms=elapsed_ms)
            if attempt > settings.max_retries:
                message = (
                    f"LLM request timed out for {config.name} after {attempt} attempt(s) "
                    f"at {settings.timeout_seconds:.1f}s"
                )
                _log_provider_failure(config, attempt=attempt, elapsed_ms=elapsed_ms, message=message)
                last_error = LLMTimeoutError(message)
                break
            last_error = LLMTimeoutError(
                f"LLM request timed out for {config.name} on attempt {attempt}"
            )
            continue
        except httpx.HTTPError as exc:
            elapsed_ms = _elapsed_ms(started_at)
            message = f"LLM request failed for {config.name}: {exc}"
            _log_provider_failure(config, attempt=attempt, elapsed_ms=elapsed_ms, message=message)
            raise LLMRequestError(message) from exc
        except ValueError as exc:
            elapsed_ms = _elapsed_ms(started_at)
            message = f"LLM response body was not valid JSON for {config.name}"
            _log_provider_failure(config, attempt=attempt, elapsed_ms=elapsed_ms, message=message)
            raise LLMResponseError(message) from exc

    if last_error is None:  # pragma: no cover - defensive guard
        raise LLMRequestError(f"LLM request failed for {config.name}")
    raise last_error


def _request_scene_intent_for_config(
    config: LLMEndpointConfig,
    prompt: str,
    scene_state: dict[str, Any],
    *,
    settings: LLMRuntimeSettings,
) -> SceneIntent:
    content = _request_content_for_config(config, _messages(prompt, scene_state, settings=settings), settings=settings)
    try:
        scene_intent = parse_scene_intent_content(content)
    except SceneIntentParseError as exc:
        raise LLMResponseError(str(exc)) from exc

    log_structured(
        _LLM_LOGGER,
        "llm_parsed_scene_intent",
        {"provider": config.name, "model": config.model, "intent": scene_intent.model_dump()},
    )
    return scene_intent


def extract_scene_intent(prompt: str, scene_state: dict[str, Any]) -> SceneIntent:
    settings = read_runtime_settings()
    last_error: LLMClientError | None = None
    for config in _provider_chain():
        try:
            return _request_scene_intent_for_config(
                config,
                prompt,
                scene_state,
                settings=settings,
            )
        except LLMClientError as exc:
            last_error = exc
            continue

    if last_error is None:  # pragma: no cover - defensive guard
        raise LLMRequestError("LLM request failed for all configured providers")
    raise last_error
