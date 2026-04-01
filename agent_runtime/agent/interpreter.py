from __future__ import annotations

import json
import re
from typing import Any

from vectra.tools.registry import ToolRegistry, get_default_registry

from .models import AgentContext, ExecutionInstruction, ReasoningStep

_CREATE_PATTERN = re.compile(
    r"^create a (?P<primitive>cube|plane|sphere|uv sphere) named (?P<name>[A-Za-z0-9_.-]+) "
    r"at \((?P<x>-?\d+(?:\.\d+)?), (?P<y>-?\d+(?:\.\d+)?), (?P<z>-?\d+(?:\.\d+)?)\)$"
)
_MOVE_PATTERN = re.compile(
    r"^move object (?P<name>[A-Za-z0-9_.-]+) to "
    r"\((?P<x>-?\d+(?:\.\d+)?), (?P<y>-?\d+(?:\.\d+)?), (?P<z>-?\d+(?:\.\d+)?)\)$"
)


def _get_registry() -> ToolRegistry:
    registry = get_default_registry()
    registry.discover()
    return registry


def _normalize_primitive(raw_value: str) -> str:
    primitive = raw_value.strip().lower()
    if primitive == "sphere":
        return "uv_sphere"
    if primitive == "uv sphere":
        return "uv_sphere"
    return primitive


def _vector_from_match(match: re.Match[str]) -> list[float]:
    return [float(match.group("x")), float(match.group("y")), float(match.group("z"))]


def _actions_from_text_steps(text_steps: list[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, step in enumerate(text_steps, start=1):
        create_match = _CREATE_PATTERN.match(step)
        if create_match:
            primitive = _normalize_primitive(create_match.group("primitive"))
            name = create_match.group("name")
            location = _vector_from_match(create_match)
            actions.append(
                {
                    "action_id": f"agent_create_{index}",
                    "tool": "mesh.create_primitive",
                    "params": {
                        "primitive_type": primitive,
                        "name": name,
                        "location": location,
                    },
                }
            )
            continue

        move_match = _MOVE_PATTERN.match(step)
        if move_match:
            actions.append(
                {
                    "action_id": f"agent_move_{index}",
                    "tool": "object.transform",
                    "params": {
                        "object_name": move_match.group("name"),
                        "location": _vector_from_match(move_match),
                    },
                }
            )
    return actions


def _actions_to_console_code(actions: list[dict[str, Any]]) -> str:
    lines = [
        "import bpy",
        "result = {'executed': [], 'created_objects': [], 'moved_objects': []}",
    ]
    created_names: dict[str, str] = {}
    for action in actions:
        tool = action.get("tool")
        params = action.get("params", {})
        action_id = action.get("action_id")
        if tool == "mesh.create_primitive":
            primitive_type = params.get("primitive_type", "cube")
            operator_name = {
                "cube": "primitive_cube_add",
                "plane": "primitive_plane_add",
                "uv_sphere": "primitive_uv_sphere_add",
            }.get(str(primitive_type), "primitive_cube_add")
            location = tuple(float(component) for component in params.get("location", [0.0, 0.0, 0.0]))
            lines.append(f"bpy.ops.mesh.{operator_name}(location={location!r})")
            if isinstance(params.get("name"), str) and params["name"].strip():
                lines.append(f"bpy.context.active_object.name = {params['name']!r}")
                lines.append(f"result['created_objects'].append({params['name']!r})")
                if isinstance(action_id, str) and action_id.strip():
                    created_names[action_id] = params["name"]
            else:
                lines.append("result['created_objects'].append(bpy.context.active_object.name)")
            lines.append(f"result['executed'].append({tool!r})")
            continue

        if tool == "object.transform":
            object_name_param = params.get("object_name", "")
            if isinstance(object_name_param, dict) and isinstance(object_name_param.get("$ref"), str):
                ref_action_id = object_name_param["$ref"].split(".", 1)[0]
                object_name = created_names.get(ref_action_id, ref_action_id)
            else:
                object_name = str(object_name_param).strip()
            lines.append(f"obj = bpy.data.objects.get({object_name!r})")
            lines.append("if obj is None:\n    raise RuntimeError('Object not found for transform')")
            if "location" in params:
                location = tuple(float(component) for component in params["location"])
                lines.append(f"obj.location = {location!r}")
            if "rotation_euler" in params:
                rotation = tuple(float(component) for component in params["rotation_euler"])
                lines.append(f"obj.rotation_euler = {rotation!r}")
            if "scale" in params:
                scale = tuple(float(component) for component in params["scale"])
                lines.append(f"obj.scale = {scale!r}")
            lines.append(f"result['moved_objects'].append({object_name!r})")
            lines.append(f"result['executed'].append({tool!r})")

    return "\n".join(lines)


def interpret_reasoning(reasoning: ReasoningStep, context: AgentContext) -> ExecutionInstruction:
    del context
    _get_registry()

    compiled_actions = reasoning.metadata.get("compiled_actions")
    if isinstance(compiled_actions, list):
        actions = [dict(action) for action in compiled_actions if isinstance(action, dict)]
    else:
        actions = _actions_from_text_steps(reasoning.intended_actions)
    console_code = reasoning.metadata.get("console_code")
    execution_metadata = reasoning.metadata.get("execution_metadata", {})

    if reasoning.status in {"clarify", "error"}:
        return ExecutionInstruction(
            kind="none",
            summary=reasoning.narration or reasoning.expected_outcome,
            expected_outcome=reasoning.expected_outcome,
            signature=reasoning.status,
            metadata=execution_metadata if isinstance(execution_metadata, dict) else {},
        )

    if not actions and (not isinstance(console_code, str) or not console_code.strip()):
        return ExecutionInstruction(
            kind="none",
            summary=reasoning.narration or reasoning.expected_outcome,
            expected_outcome=reasoning.expected_outcome,
            signature="none",
            metadata=execution_metadata if isinstance(execution_metadata, dict) else {},
        )

    if reasoning.preferred_execution_mode == "vectra-code":
        code = console_code
        if not isinstance(code, str) or not code.strip():
            code = _actions_to_console_code(actions)
        signature = json.dumps({"kind": "console_code", "code": code}, sort_keys=True)
        return ExecutionInstruction(
            kind="console_code",
            summary=reasoning.narration,
            actions=actions,
            code=code,
            expected_outcome=reasoning.expected_outcome,
            signature=signature,
            metadata=execution_metadata if isinstance(execution_metadata, dict) else {},
        )

    signature = json.dumps({"kind": "tool_actions", "actions": actions}, sort_keys=True)
    return ExecutionInstruction(
        kind="tool_actions",
        summary=reasoning.narration,
        actions=actions,
        expected_outcome=reasoning.expected_outcome,
        signature=signature,
        metadata=execution_metadata if isinstance(execution_metadata, dict) else {},
    )
