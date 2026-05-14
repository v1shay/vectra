from __future__ import annotations

import re
from typing import Any

from .models import DirectorContext, ToolCall

_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "in",
    "into",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
}


def _semantic_terms(prompt: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", prompt.lower()):
        if token in _STOPWORDS or token in terms:
            continue
        terms.append(token)
    return terms[:16]


def _prompt_obligations(prompt: str) -> list[str]:
    chunks = [
        " ".join(chunk.split())
        for chunk in re.split(r"[.;:\n]+|\bthen\b|\band\b", prompt)
        if " ".join(chunk.split())
    ]
    if not chunks:
        normalized = " ".join(prompt.split())
        return [normalized] if normalized else []
    return chunks[:12]


def _scene_nodes(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    raw_objects = scene_state.get("objects", [])
    if not isinstance(raw_objects, list):
        return []
    nodes: list[dict[str, Any]] = []
    for raw_object in raw_objects[:24]:
        if not isinstance(raw_object, dict):
            continue
        name = raw_object.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        nodes.append(
            {
                "name": name,
                "type": raw_object.get("type") or raw_object.get("kind") or "OBJECT",
                "location": raw_object.get("location"),
                "dimensions": raw_object.get("dimensions"),
            }
        )
    return nodes


def _planned_output_nodes(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for action in actions:
        params = action.get("params", {})
        if not isinstance(params, dict):
            continue
        name = params.get("name") or params.get("object_name") or params.get("target")
        if isinstance(name, dict):
            name = name.get("$ref")
        if not isinstance(name, str) or not name.strip():
            continue
        nodes.append(
            {
                "name": name,
                "source_action": action.get("action_id"),
                "tool": action.get("tool"),
                "location": params.get("location"),
            }
        )
    return nodes[:24]


def _task_graph(tool_calls: list[ToolCall], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls, start=1):
        action = actions[index - 1] if index - 1 < len(actions) else {}
        tasks.append(
            {
                "id": f"step_{index}",
                "tool": tool_call.name,
                "action_id": action.get("action_id"),
                "depends_on": [f"step_{index - 1}"] if index > 1 else [],
                "parameters": sorted(tool_call.arguments.keys()),
            }
        )
    return tasks


def build_organic_scene_metadata(
    *,
    context: DirectorContext,
    tool_calls: list[ToolCall],
    actions: list[dict[str, Any]],
    action_families: list[str],
) -> dict[str, Any]:
    return {
        "planning_mode": "organic_scene_graph_v1",
        "hardcoding_policy": "clean",
        "semantic_terms": _semantic_terms(context.user_prompt),
        "prompt_obligations": _prompt_obligations(context.user_prompt),
        "organic_scene_graph": {
            "existing_nodes": _scene_nodes(context.scene_state),
            "planned_nodes": _planned_output_nodes(actions),
        },
        "organic_task_graph": _task_graph(tool_calls, actions),
        "tool_families_used": list(dict.fromkeys(action_families)),
    }
