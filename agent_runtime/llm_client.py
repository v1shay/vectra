from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from vectra.tools.registry import get_default_registry

DEFAULT_TIMEOUT_SECONDS = 20.0
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


class LLMResponseError(LLMClientError):
    """Raised when the LLM response is invalid."""


@dataclass(frozen=True)
class LLMEndpointConfig:
    name: str
    base_url: str
    api_key: str
    model: str


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
    return LLMEndpointConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
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
        response = httpx.get(
            f"{root_url}/api/tags",
            timeout=OLLAMA_DISCOVERY_TIMEOUT_SECONDS,
        )
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
    return LLMEndpointConfig(
        name="ollama",
        base_url=f"{root_url}/v1",
        api_key=api_key,
        model=model,
    )


def _available_configs() -> list[LLMEndpointConfig]:
    configs = [
        _read_env_config(
            name="primary",
            base_url_var="VECTRA_LLM_BASE_URL",
            api_key_var="VECTRA_LLM_API_KEY",
            model_var="VECTRA_LLM_MODEL",
        ),
        _read_env_config(
            name="fallback",
            base_url_var="VECTRA_LLM_FALLBACK_BASE_URL",
            api_key_var="VECTRA_LLM_FALLBACK_API_KEY",
            model_var="VECTRA_LLM_FALLBACK_MODEL",
        ),
        _read_ollama_config(),
    ]
    available = [config for config in configs if config is not None]
    if available:
        return available

    raise LLMConfigurationError(
        "Missing LLM configuration. Set VECTRA_LLM_* vars, or VECTRA_LLM_FALLBACK_* vars, "
        "or run a local Ollama model."
    )


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


def _system_prompt() -> str:
    tool_metadata = json.dumps(_tool_metadata(), indent=2, sort_keys=True)
    return (
        "You are Vectra's semantic planner.\n"
        "Return only a JSON array of actions.\n"
        "Do not return markdown.\n"
        "Do not explain your reasoning.\n"
        "Each action must be an object with:\n"
        '- optional "action_id" (unique string if present)\n'
        '- required "tool" (must exactly match one listed tool name)\n'
        '- required "params" (JSON object)\n'
        'Reference syntax must use JSON object form {"$ref": "action_id.output_key"}.\n'
        "Never emit $ref(...) strings or any other shorthand.\n"
        "Use the minimum number of actions needed.\n"
        "Do not hallucinate tools, params, or outputs.\n"
        "You may use $ref chaining only when the referenced action_id and output key exist in the listed tool output_schema.\n"
        "If the request cannot be completed with the available tools, return [].\n"
        "Available tools:\n"
        f"{tool_metadata}"
    )


def _messages(prompt: str, scene_state: dict[str, Any], correction: str | None = None) -> list[dict[str, str]]:
    payload = json.dumps({"prompt": prompt, "scene_state": scene_state}, indent=2, sort_keys=True)
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": payload},
    ]
    if correction is not None:
        messages.append({"role": "user", "content": correction})
    return messages


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


def _strip_markdown_fences(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_actions(content: str) -> list[dict[str, Any]]:
    normalized = _strip_markdown_fences(content)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, list):
        raise LLMResponseError("LLM response must be a JSON array")
    return parsed


def _request_actions_for_config(
    config: LLMEndpointConfig,
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
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

    return _parse_actions(_extract_message_content(payload))


def _request_actions(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    last_error: LLMClientError | None = None
    for config in _available_configs():
        try:
            return _request_actions_for_config(config, messages)
        except LLMClientError as exc:
            last_error = exc

    if last_error is None:  # pragma: no cover - defensive guard
        raise LLMRequestError("LLM request failed for all configured providers")
    raise last_error


def generate_actions(prompt: str, scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    messages = _messages(prompt, scene_state)
    try:
        return _request_actions(messages)
    except LLMResponseError:
        retry_messages = _messages(
            prompt,
            scene_state,
            correction=(
                "Your previous response was invalid. "
                "Return only a raw JSON array of actions with no markdown fences or explanations."
            ),
        )
        return _request_actions(retry_messages)
