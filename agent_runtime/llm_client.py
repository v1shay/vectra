from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from vectra.tools.registry import get_default_registry
from vectra.utils.logging import get_vectra_logger, log_structured

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
    return LLMEndpointConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


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
    primary = _primary_config()
    fallback = _read_env_config(
        name="fallback",
        base_url_var="VECTRA_LLM_FALLBACK_BASE_URL",
        api_key_var="VECTRA_LLM_FALLBACK_API_KEY",
        model_var="VECTRA_LLM_FALLBACK_MODEL",
    )
    ollama = _read_ollama_config()
    available = [config for config in (primary, fallback, ollama) if config is not None]
    if available:
        return available

    raise LLMConfigurationError(
        "Missing LLM configuration. Set VECTRA_LLM_* vars or run a local Ollama model."
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


def _format_examples() -> str:
    examples = [
        {
            "name": "Create cube at origin",
            "scene": {
                "active_object": None,
                "selected_objects": [],
                "objects": [],
            },
            "user": "create a cube",
            "output": [
                {
                    "action_id": "create_cube",
                    "tool": "mesh.create_primitive",
                    "params": {
                        "primitive_type": "cube",
                        "location": [0, 0, 0],
                    },
                }
            ],
        },
        {
            "name": "Move forward on +Y",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move cube forward 2",
            "output": [
                {
                    "action_id": "move_cube_forward",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [0, 2, 0],
                    },
                }
            ],
        },
        {
            "name": "Colloquial reference can still resolve to the active object",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move this shit forward 2",
            "output": [
                {
                    "action_id": "move_active_forward",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [0, 2, 0],
                    },
                }
            ],
        },
        {
            "name": "Very informal shorthand can still refer to the active object when unique",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move ts forward 2",
            "output": [
                {
                    "action_id": "move_active_forward",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [0, 2, 0],
                    },
                }
            ],
        },
        {
            "name": "Move backward on -Y",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move cube back 2",
            "output": [
                {
                    "action_id": "move_cube_back",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [0, -2, 0],
                    },
                }
            ],
        },
        {
            "name": "Move upward on +Z",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move cube up 3",
            "output": [
                {
                    "action_id": "move_cube_up",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [0, 0, 3],
                    },
                }
            ],
        },
        {
            "name": "Move downward on -Z",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move cube down 1",
            "output": [
                {
                    "action_id": "move_cube_down",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [0, 0, -1],
                    },
                }
            ],
        },
        {
            "name": "Unspecified rotation uses Z axis",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "rotation_euler": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "rotate cube 45 degrees",
            "output": [
                {
                    "action_id": "rotate_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "rotation_euler": [0, 0, 0.785398],
                    },
                }
            ],
        },
        {
            "name": "Use $ref only when a later action depends on an earlier output",
            "scene": {
                "active_object": None,
                "selected_objects": [],
                "objects": [],
            },
            "user": "create a cube and move it forward 2",
            "output": [
                {
                    "action_id": "create_cube",
                    "tool": "mesh.create_primitive",
                    "params": {
                        "primitive_type": "cube",
                        "location": [0, 0, 0],
                    },
                },
                {
                    "action_id": "move_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": {"$ref": "create_cube.object_name"},
                        "location": [0, 2, 0],
                    },
                },
            ],
        },
        {
            "name": "Ambiguous language must fail instead of guessing",
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [0, 0, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "move cube somewhere weird",
            "output": [],
        },
        {
            "name": "Vague creation requests must fail instead of inventing geometry",
            "scene": {
                "active_object": None,
                "selected_objects": [],
                "objects": [],
            },
            "user": "make some shit",
            "output": [],
        },
    ]
    rendered: list[str] = []
    for example in examples:
        rendered.extend(
            [
                f"Example: {example['name']}",
                f"- Scene: {json.dumps(example['scene'], sort_keys=True)}",
                f"- User: {example['user']}",
                f"- Valid output: {json.dumps(example['output'])}",
            ]
        )
    return "\n".join(rendered)


def _system_prompt() -> str:
    tool_catalog = _format_tool_catalog()
    examples = _format_examples()
    return (
        "You are Vectra's semantic planner.\n"
        "Convert the user request and scene_state into a valid JSON array of structured actions.\n"
        "Return JSON array only. Do not return markdown. Do not explain. Do not add any text before or after the JSON.\n"
        "Each action object must contain exactly these keys:\n"
        '- optional "action_id" (unique non-empty string when present)\n'
        '- required "tool"\n'
        '- required "params"\n'
        "No extra top-level keys are allowed.\n"
        "Tool names must exactly match the tool catalog.\n"
        "Params must match the tool schema exactly: no extra params, no missing required params, no nulls, no invented fields.\n"
        "Spatial coordinate system is fixed and global:\n"
        "- +X = right\n"
        "- -X = left\n"
        "- +Y = forward\n"
        "- -Y = backward\n"
        "- +Z = up\n"
        "- -Z = down\n"
        "Vector params must be explicit JSON arrays in [x, y, z] order.\n"
        "Do not use axis maps like {\"x\": 1}. Use full arrays only.\n"
        "If the user gives only one directional move, preserve the object's other coordinates from scene_state and output the final full vector.\n"
        "If the user says rotate without an axis, use the Z axis.\n"
        "Angles in rotation_euler must be radians.\n"
        "Colloquial references like 'it', 'this', 'that', 'this thing', 'that thing', 'this object', "
        "'this cube', 'this shit', and shorthand like 'ts' refer to the active object or only selected "
        "object when the reference is unique in scene_state.\n"
        "Profanity, slang, and casual phrasing do not change the underlying action semantics.\n"
        "Use [] when the request is ambiguous, unsupported, or cannot be grounded safely from scene_state.\n"
        "If the user does not specify a concrete operation or target, return [].\n"
        "Never guess hidden intent. Never fabricate coordinates for vague language like 'somewhere weird'.\n"
        "Use $ref only in the exact form {\"$ref\":\"action_id.output_key\"}.\n"
        "Only use $ref when the referenced action_id already exists earlier in the same array and the output_key exists in that tool's output schema.\n"
        "If a user refers to an existing object, resolve it from scene_state.objects. Prefer exact name match, then active object, then the only selected object.\n"
        "Tool catalog:\n"
        f"{tool_catalog}\n"
        "Examples:\n"
        f"{examples}"
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


def _parse_actions(content: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, list):
        raise LLMResponseError("LLM response must be a JSON array")
    return parsed


def _request_content_for_config(
    config: LLMEndpointConfig,
    messages: list[dict[str, str]],
) -> str:
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
    log_structured(
        _LLM_LOGGER,
        "llm_raw_output",
        {"provider": config.name, "content": content},
    )
    return content


def _request_actions_for_config(
    config: LLMEndpointConfig,
    prompt: str,
    scene_state: dict[str, Any],
) -> list[dict[str, Any]]:
    content = _request_content_for_config(config, _messages(prompt, scene_state))
    actions = _parse_actions(content)
    log_structured(
        _LLM_LOGGER,
        "llm_parsed_json",
        {"provider": config.name, "actions": actions},
    )
    return actions


def _request_actions(prompt: str, scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    last_error: LLMClientError | None = None
    for config in _available_configs():
        try:
            return _request_actions_for_config(config, prompt, scene_state)
        except LLMClientError as exc:
            last_error = exc

    if last_error is None:  # pragma: no cover - defensive guard
        raise LLMRequestError("LLM request failed for all configured providers")
    raise last_error


def generate_actions(prompt: str, scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    return _request_actions(prompt, scene_state)
