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


def _format_tool_catalog() -> str:
    lines: list[str] = []
    for tool in _tool_metadata():
        lines.extend(
            [
                f"Tool: {tool['name']}",
                f"- Meaning: {tool['description']}",
                f"- Input schema: {json.dumps(tool['input_schema'], sort_keys=True)}",
                f"- Output schema: {json.dumps(tool['output_schema'], sort_keys=True)}",
            ]
        )
    return "\n".join(lines)


def _format_examples() -> str:
    examples = [
        {
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
            "user": "move cube to x 10",
            "output": [
                {
                    "action_id": "move_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [10, 0, 0],
                    },
                }
            ],
            "why": "Use the existing Cube and update only the x coordinate while keeping the other coordinates from the current scene state.",
        },
        {
            "scene": {
                "active_object": "Cube",
                "selected_objects": ["Cube"],
                "objects": [
                    {
                        "name": "Cube",
                        "location": [3, 1, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "shift the cube right",
            "output": [
                {
                    "action_id": "shift_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [5, 1, 0],
                    },
                }
            ],
            "why": "Right means a positive x move. When no distance is given, use a default relative step of 2 units and preserve y and z.",
        },
        {
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
            "user": "put the cube at position 10 on x axis",
            "output": [
                {
                    "action_id": "place_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [10, 0, 0],
                    },
                }
            ],
            "why": "An absolute x-axis instruction becomes a full location vector that keeps the current y and z values.",
        },
        {
            "scene": {
                "active_object": "Sphere",
                "selected_objects": ["Sphere"],
                "objects": [
                    {
                        "name": "Sphere",
                        "location": [1, 2, 0],
                        "selected": True,
                        "active": True,
                    }
                ],
            },
            "user": "slide it over",
            "output": [
                {
                    "action_id": "slide_active_object",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Sphere",
                        "location": [3, 2, 0],
                    },
                }
            ],
            "why": "Resolve 'it' to the active object, then apply a relative move using the current coordinates from scene state.",
        },
        {
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
            "user": "put cube at 5 0 0",
            "output": [
                {
                    "action_id": "put_cube",
                    "tool": "object.transform",
                    "params": {
                        "object_name": "Cube",
                        "location": [5, 0, 0],
                    },
                }
            ],
            "why": "A three-number position request becomes a full absolute location vector.",
        },
    ]
    rendered: list[str] = []
    for index, example in enumerate(examples, start=1):
        rendered.extend(
            [
                f"Example {index}",
                f"- Scene: {json.dumps(example['scene'], sort_keys=True)}",
                f"- User: {example['user']}",
                f"- Valid output: {json.dumps(example['output'])}",
                f"- Why valid: {example['why']}",
            ]
        )
    return "\n".join(rendered)


def _system_prompt() -> str:
    tool_catalog = _format_tool_catalog()
    examples = _format_examples()
    return (
        "You are Vectra's semantic planner.\n"
        "Your job is to convert a natural-language request plus scene state into a valid JSON array of structured actions.\n"
        "You are not writing Python and you are not describing what to do in prose.\n"
        "Return only a JSON array of actions.\n"
        "Do not return markdown.\n"
        "Do not explain your reasoning.\n"
        "Each action must be an object with:\n"
        '- optional "action_id" (unique string if present)\n'
        '- required "tool" (must exactly match one listed tool name)\n'
        '- required "params" (JSON object)\n'
        "Use the fewest actions needed.\n"
        "Prefer a valid action grounded in the available tools and scene state over returning an empty plan for a simple request.\n"
        "Only return [] if the request truly cannot be expressed with the available tools.\n"
        'Reference syntax must use JSON object form {"$ref": "action_id.output_key"}.\n'
        "Never emit $ref(...) strings or any other shorthand.\n"
        "Do not hallucinate tools, params, or outputs.\n"
        "You may use $ref chaining only when the referenced action_id and output key exist in the listed tool output_schema.\n"
        "Use scene_state.objects to resolve existing objects.\n"
        "Prefer the active object first, then selected objects, then the closest exact scene object name.\n"
        "If the user says 'cube', and an object named 'Cube' exists in scene_state.objects, use 'Cube' as object_name.\n"
        "If the user says 'it', resolve 'it' to the active object first, then the only selected object.\n"
        "location, rotation_euler, and scale are vectors in [x, y, z] order.\n"
        "If the user specifies only one axis, preserve the other axes from the matched object's current transform in scene_state.objects.\n"
        "For example, 'x 10' means [10, current_y, current_z].\n"
        "If the user asks for a relative move like right, left, up, down, forward, or back without a number, convert it into a concrete vector by changing only the implied axis and using a default relative step of 2 Blender units.\n"
        "A request to move or shift an existing object should usually use object.transform.\n"
        "A request to create a new basic shape should usually use mesh.create_primitive.\n"
        "Choose the closest supported primitive only when it matches the user's requested shape and no existing object is being targeted.\n"
        "Tool catalog:\n"
        f"{tool_catalog}\n"
        "Illustrative examples. These teach the representation pattern and are not an exhaustive list of accepted phrases:\n"
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


def _messages(
    prompt: str,
    scene_state: dict[str, Any],
    *,
    invalid_response: str | None = None,
    repair: bool = False,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_content(prompt, scene_state)},
    ]
    if invalid_response is not None:
        messages.append({"role": "assistant", "content": invalid_response})
    if repair:
        messages.append(
            {
                "role": "user",
                "content": "Return ONLY valid JSON array. No explanation.",
            }
        )
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
    print("RAW LLM OUTPUT:", content)
    return content


def _request_actions_for_config(
    config: LLMEndpointConfig,
    prompt: str,
    scene_state: dict[str, Any],
) -> list[dict[str, Any]]:
    messages = _messages(prompt, scene_state)
    content = _request_content_for_config(config, messages)
    try:
        return _parse_actions(content)
    except LLMResponseError as first_exc:
        repair_messages = _messages(
            prompt,
            scene_state,
            invalid_response=content,
            repair=True,
        )
        repair_content = _request_content_for_config(config, repair_messages)
        try:
            return _parse_actions(repair_content)
        except LLMResponseError as repair_exc:
            raise LLMResponseError(
                f"{first_exc}; repair failed: {repair_exc}"
            ) from repair_exc


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
