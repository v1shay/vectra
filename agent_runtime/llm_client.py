from __future__ import annotations

import json
import os
from typing import Any

import httpx

from vectra.tools.registry import get_default_registry

DEFAULT_TIMEOUT_SECONDS = 20.0


class LLMClientError(Exception):
    """Base exception for LLM planning client failures."""


class LLMConfigurationError(LLMClientError):
    """Raised when required LLM configuration is missing."""


class LLMRequestError(LLMClientError):
    """Raised when the LLM request cannot be completed."""


class LLMResponseError(LLMClientError):
    """Raised when the LLM response is invalid."""


def _read_config() -> tuple[str, str, str]:
    base_url = os.getenv("VECTRA_LLM_BASE_URL", "").strip()
    api_key = os.getenv("VECTRA_LLM_API_KEY", "").strip()
    model = os.getenv("VECTRA_LLM_MODEL", "").strip()
    missing = [
        env_name
        for env_name, value in (
            ("VECTRA_LLM_BASE_URL", base_url),
            ("VECTRA_LLM_API_KEY", api_key),
            ("VECTRA_LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise LLMConfigurationError(f"Missing LLM configuration: {', '.join(missing)}")
    return base_url.rstrip("/"), api_key, model


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


def _request_actions(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    base_url, api_key, model = _read_config()
    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": messages,
            },
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"LLM request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMResponseError("LLM response body was not valid JSON") from exc

    return _parse_actions(_extract_message_content(payload))


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
