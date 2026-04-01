from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from vectra.tools.registry import get_default_registry
from vectra.utils.logging import get_vectra_logger, log_structured

try:
    from .intent import IntentEnvelope
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from intent import IntentEnvelope

DEFAULT_TIMEOUT_SECONDS = 20.0
OLLAMA_DISCOVERY_TIMEOUT_SECONDS = 2.0
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_API_KEY = "ollama"
DEFAULT_PROVIDER_CONFIDENCE = 0.35
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


class LLMResponseError(LLMClientError):
    """Raised when the LLM response is invalid."""


@dataclass(frozen=True)
class LLMEndpointConfig:
    name: str
    base_url: str
    api_key: str
    model: str


_LLM_LOGGER = get_vectra_logger("vectra.runtime.llm")


def _normalize_http_url(raw_url: str) -> str:
    normalized = raw_url.strip()
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


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


def _tool_metadata() -> list[dict[str, Any]]:
    registry = get_default_registry()
    registry.discover()
    metadata: list[dict[str, Any]] = []
    for tool_name in registry.list_tools():
        tool = registry.get(tool_name)
        metadata.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
        )
    return metadata


def _format_tool_catalog() -> str:
    lines: list[str] = []
    for tool in _tool_metadata():
        lines.extend(
            [
                f"Tool: {tool['name']}",
                f"- Meaning: {tool['description']}",
                f"- Exact input schema: {json.dumps(tool['input_schema'], sort_keys=True)}",
                f"- Exact output schema: {json.dumps(tool['output_schema'], sort_keys=True)}",
            ]
        )
    return "\n".join(lines)


def _intent_schema_description() -> str:
    return json.dumps(
        {
            "status": "ok | no_action",
            "confidence": "number between 0 and 1",
            "reason": "string",
            "steps": [
                {
                    "action": "create | transform",
                    "target": "optional raw target phrase",
                    "target_name": "optional exact object name when certain",
                    "target_ref": (
                        "optional reference type such as active_object, selected_object, "
                        "previous_step"
                    ),
                    "primitive_type": "optional primitive name for create intents",
                    "direction": "optional direction label",
                    "magnitude": "optional numeric magnitude",
                    "magnitude_qualifier": "optional vague magnitude phrase such as 'a bit' or 'a couple'",
                    "transform_kind": "optional location | rotation | scale",
                    "axis": "optional x | y | z",
                    "confidence": "number between 0 and 1",
                }
            ],
        },
        indent=2,
        sort_keys=True,
    )


def _system_prompt() -> str:
    tool_catalog = _format_tool_catalog()
    return (
        "You are Vectra's intent parser.\n"
        "Convert the user request and scene_state into a high-level JSON intent object.\n"
        "Return JSON object only. Do not return markdown. Do not explain. Do not produce actions.\n"
        "The output must exactly match the documented intent schema.\n"
        "Use status='no_action' with an empty steps list when the request is ambiguous, unsupported, "
        "or unsafe to ground from scene_state.\n"
        "Do not guess missing targets, directions, tools, or geometry.\n"
        "Normalize requests into a compact semantic form:\n"
        "- action=create only when the primitive is explicit and supported\n"
        "- action=transform for movement, rotation, or scale adjustments\n"
        "- transform_kind should be one of location, rotation, or scale\n"
        "- direction should capture movement semantics like forward, backward, up, down, left, right\n"
        "- if the user says 'into the ground', preserve that as a downward movement intent\n"
        "- if the user gives vague magnitudes like 'a bit' or 'a couple', store them in magnitude_qualifier\n"
        "- if the user refers to the active or selected object indirectly, use target_ref when appropriate\n"
        "- if a later step refers to an object created earlier in the same request, use target_ref='previous_step'\n"
        "- for rotation without an axis, prefer axis='z'\n"
        "Only describe intent. The local planner will decide exact actions.\n"
        "Supported tool catalog for downstream planning:\n"
        f"{tool_catalog}\n"
        "Intent schema:\n"
        f"{_intent_schema_description()}"
    )


def _scene_object_summary(scene_state: dict[str, Any]) -> str:
    objects = scene_state.get("objects", [])
    if not isinstance(objects, list) or not objects:
        return "- No scene objects provided"

    lines: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        lines.append(
            "- "
            f"{obj.get('name', '<unknown>')} | type={obj.get('type', '<unknown>')} | "
            f"active={obj.get('active', False)} | selected={obj.get('selected', False)} | "
            f"location={obj.get('location', [])} | rotation_euler={obj.get('rotation_euler', [])} | "
            f"scale={obj.get('scale', [])}"
        )
    return "\n".join(lines) if lines else "- No scene objects provided"


def _user_content(prompt: str, scene_state: dict[str, Any]) -> str:
    return (
        f"User request:\n{prompt}\n\n"
        "Scene focus:\n"
        f"- active_object: {scene_state.get('active_object')}\n"
        f"- selected_objects: {scene_state.get('selected_objects', [])}\n"
        f"- current_frame: {scene_state.get('current_frame')}\n\n"
        "Scene objects summary:\n"
        f"{_scene_object_summary(scene_state)}\n\n"
        "Raw scene_state JSON:\n"
        f"{json.dumps(scene_state, indent=2, sort_keys=True)}"
    )


def _messages(prompt: str, scene_state: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_content(prompt, scene_state)},
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


def _parse_intent(content: str) -> IntentEnvelope:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise LLMResponseError("LLM response must be a JSON object")

    try:
        return IntentEnvelope.model_validate(parsed)
    except Exception as exc:
        raise LLMResponseError(f"LLM response did not match the intent schema: {exc}") from exc


def _request_content_for_config(
    config: LLMEndpointConfig,
    messages: list[dict[str, str]],
) -> str:
    log_structured(_LLM_LOGGER, "llm_provider_attempt", {"provider": config.name, "model": config.model})
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
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"LLM request failed for {config.name}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMResponseError(f"LLM response body was not valid JSON for {config.name}") from exc

    content = _extract_message_content(payload)
    log_structured(_LLM_LOGGER, "llm_provider_used", {"provider": config.name, "model": config.model})
    log_structured(_LLM_LOGGER, "llm_raw_output", {"provider": config.name, "content": content})
    return content


def _request_intent_for_config(
    config: LLMEndpointConfig,
    prompt: str,
    scene_state: dict[str, Any],
) -> IntentEnvelope:
    content = _request_content_for_config(config, _messages(prompt, scene_state))
    intent = _parse_intent(content)
    log_structured(
        _LLM_LOGGER,
        "llm_parsed_intent",
        {"provider": config.name, "intent": intent.model_dump()},
    )
    return intent


def extract_intent(prompt: str, scene_state: dict[str, Any]) -> IntentEnvelope:
    last_error: LLMClientError | None = None
    for index, config in enumerate(_provider_chain()):
        try:
            return _request_intent_for_config(config, prompt, scene_state)
        except LLMClientError as exc:
            last_error = exc
            is_primary = index == 0
            if is_primary:
                log_structured(
                    _LLM_LOGGER,
                    "llm_provider_failure",
                    {"provider": config.name, "message": str(exc)},
                    level="warning",
                )
                continue
            break

    if last_error is None:  # pragma: no cover - defensive guard
        raise LLMRequestError("LLM request failed for all configured providers")
    raise last_error
