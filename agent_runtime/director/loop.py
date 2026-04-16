from __future__ import annotations

import json
from typing import Any

from vectra.tools.base import BaseTool, ToolValidationError
from vectra.tools.registry import ToolNotFoundError, ToolRegistry, get_default_registry
from vectra.utils.logging import get_vectra_logger, log_structured

from .models import (
    AssumptionRecord,
    BudgetState,
    ControllerDecision,
    DirectorContext,
    DirectorTurn,
    ProviderResult,
    ToolCall,
    ToolCallValidationIssue,
)
from .providers import ProviderError, call_controller, call_director
from .resolver import ReferenceResolver

_TURN_BUDGETS = {"low": 8, "medium": 12, "high": 16}
_BATCH_LIMIT = 4
_LOGGER = get_vectra_logger("vectra.runtime.director.loop")


def _json_schema_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    spec_type = spec.get("type", "string")
    if spec_type == "string":
        schema: dict[str, Any] = {"type": "string"}
        if isinstance(spec.get("enum"), list):
            schema["enum"] = list(spec["enum"])
        return schema
    if spec_type == "integer":
        return {"type": "integer"}
    if spec_type == "number":
        return {"type": "number"}
    if spec_type == "boolean":
        return {"type": "boolean"}
    if spec_type == "vector3":
        return {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        }
    if spec_type == "string_array":
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "object"}


def _tool_to_schema(tool: BaseTool) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in tool.input_schema.items():
        if isinstance(spec, dict):
            properties[name] = _json_schema_for_spec(spec)
            description = spec.get("description")
            if isinstance(description, str) and description.strip():
                properties[name]["description"] = description.strip()
            if spec.get("required"):
                required.append(name)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _control_tools(execution_mode: str) -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "name": "task.complete",
            "description": "Use when the current user request is satisfied and no further action is needed.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
        {
            "type": "function",
            "name": "task.clarify",
            "description": "Use only when there is no safe or physically possible way to continue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["question", "reason"],
            },
        },
    ]
    if execution_mode == "vectra-code":
        tools.append(
            {
                "type": "function",
                "name": "python.execute_blender_snippet",
                "description": "Execute a short Blender Python snippet only when an atomic tool cannot express the step.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        )
    return tools


def _tool_family(tool_name: str) -> str:
    if tool_name.startswith("mesh.create"):
        return "create"
    if tool_name.startswith("object.transform"):
        return "transform"
    if tool_name.startswith("object.place_") or tool_name == "object.align_to":
        return "layout"
    if tool_name.startswith("object.duplicate"):
        return "duplicate"
    if tool_name.startswith("object.delete"):
        return "delete"
    if tool_name.startswith("object.distribute") or tool_name.startswith("object.align"):
        return "layout"
    if tool_name.startswith("object.parent") or tool_name.startswith("scene.group"):
        return "structure"
    if tool_name == "scene.ensure_floor":
        return "structure"
    if tool_name.startswith("material."):
        return "material"
    if tool_name.startswith("light."):
        return "light"
    if tool_name.startswith("camera."):
        return "camera"
    if tool_name.startswith("scene.set_frame") or tool_name.startswith("object.keyframe"):
        return "animation"
    if tool_name.startswith("scene.capture") or tool_name.startswith("scene.get_state") or tool_name.startswith("scene.frame_view"):
        return "observe"
    if tool_name.startswith("python."):
        return "code"
    return "generic"


def _drop_none_values(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _compact_scene_state(scene_state: dict[str, Any]) -> str:
    objects = scene_state.get("objects", [])
    compact_objects: list[str] = []
    if isinstance(objects, list):
        for obj in objects[:24]:
            if not isinstance(obj, dict):
                continue
            spatial = obj.get("spatial", {}) if isinstance(obj.get("spatial"), dict) else {}
            relations = obj.get("relations", []) if isinstance(obj.get("relations"), list) else []
            relation_text = ", ".join(
                f"{item.get('relation')}:{item.get('target')}"
                for item in relations[:6]
                if isinstance(item, dict)
            )
            compact_objects.append(
                "name={name} type={type} active={active} selected={selected} "
                "location={location} dimensions={dimensions} bounds={bounds} "
                "spatial_center={spatial_center} grounded={grounded} floor_contact={floor_contact} "
                "wall_like={wall_like} floor_like={floor_like} relations={relations} "
                "parent={parent} materials={materials} animation={animation}".format(
                    name=obj.get("name"),
                    type=obj.get("type"),
                    active=obj.get("active", False),
                    selected=obj.get("selected", False),
                    location=obj.get("location"),
                    dimensions=obj.get("dimensions"),
                    bounds=obj.get("bounds"),
                    spatial_center=spatial.get("center"),
                    grounded=spatial.get("grounded"),
                    floor_contact=spatial.get("floor_contact"),
                    wall_like=spatial.get("is_wall_like"),
                    floor_like=spatial.get("is_floor_like"),
                    relations=relation_text or "none",
                    parent=obj.get("parent"),
                    materials=obj.get("material_names", []),
                    animation=obj.get("keyframe_count", 0),
                )
            )
    lights = scene_state.get("lights", [])
    groups = scene_state.get("groups", [])
    return (
        f"Active object: {scene_state.get('active_object')}\n"
        f"Selected objects: {scene_state.get('selected_objects', [])}\n"
        f"Active camera: {scene_state.get('active_camera')}\n"
        f"Current frame: {scene_state.get('current_frame')}\n"
        f"Scene centroid: {scene_state.get('scene_centroid')}\n"
        f"Scene bounds: {scene_state.get('scene_bounds')}\n"
        f"Lights: {lights}\n"
        f"Groups: {groups}\n"
        f"Objects:\n- " + "\n- ".join(compact_objects)
        if compact_objects
        else (
            f"Active object: {scene_state.get('active_object')}\n"
            f"Selected objects: {scene_state.get('selected_objects', [])}\n"
            f"Active camera: {scene_state.get('active_camera')}\n"
            f"Current frame: {scene_state.get('current_frame')}\n"
            f"Scene centroid: {scene_state.get('scene_centroid')}\n"
            f"Scene bounds: {scene_state.get('scene_bounds')}\n"
            f"Lights: {lights}\n"
            f"Groups: {groups}\n"
            "Objects: none"
        )
    )


def _history_summary(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No prior execution history."
    snippets: list[str] = []
    for entry in history[-8:]:
        if not isinstance(entry, dict):
            continue
        snippets.append(f"[{entry.get('role', 'unknown')}] {entry.get('summary', '')}")
    return "\n".join(snippets) if snippets else "No prior execution history."


def _memory_summary(memory_results: list[dict[str, Any]]) -> str:
    if not memory_results:
        return "No relevant memory."
    return "\n".join(str(record.get("summary", "")) for record in memory_results[:5])


def _latest_observation_summary(context: DirectorContext) -> str:
    if context.latest_observation is not None:
        return context.latest_observation.summary
    for entry in reversed(context.history):
        if isinstance(entry, dict) and entry.get("role") == "verification":
            return str(entry.get("summary", "No fresh observation summary yet.")).strip()
    return "No fresh observation summary yet."


def _core_task_started(history: list[dict[str, Any]]) -> bool:
    for entry in history:
        if not isinstance(entry, dict) or entry.get("role") != "verification":
            continue
        details = entry.get("details", {})
        if not isinstance(details, dict):
            continue
        if bool(details.get("meaningful_change")):
            return True
        if details.get("created_objects") or details.get("moved_objects") or details.get("changed_objects"):
            return True
    return False


def _derive_budget_state(context: DirectorContext, decision: ControllerDecision) -> BudgetState:
    complexity = decision.complexity if decision.complexity in _TURN_BUDGETS else "medium"
    turn_budget = _TURN_BUDGETS[complexity]
    turns_used = max(min(context.iteration - 1, turn_budget), 0)
    turns_remaining = max(turn_budget - turns_used, 0)
    completion_mode_active = turns_used >= max(int(turn_budget * 0.75), 1)
    return BudgetState(
        complexity=complexity,
        turn_budget=turn_budget,
        turns_used=turns_used,
        turns_remaining=turns_remaining,
        completion_mode_active=completion_mode_active,
        core_task_started=_core_task_started(context.history),
    )


def _build_director_prompt(
    context: DirectorContext,
    decision: ControllerDecision,
    budget_state: BudgetState,
) -> str:
    screenshot = context.screenshot or {}
    controller_hint = decision.raw or {
        "needs_scene_context": decision.needs_scene_context,
        "needs_visual_feedback": decision.needs_visual_feedback,
        "complexity": decision.complexity,
    }
    return (
        f"User prompt:\n{context.user_prompt}\n\n"
        f"Turn budget state:\n{json.dumps({'complexity': budget_state.complexity, 'turn_budget': budget_state.turn_budget, 'turns_used': budget_state.turns_used, 'turns_remaining': budget_state.turns_remaining, 'completion_mode_active': budget_state.completion_mode_active, 'core_task_started': budget_state.core_task_started}, indent=2)}\n\n"
        f"Controller hints:\n{json.dumps(controller_hint, indent=2)}\n\n"
        f"Scene state:\n{_compact_scene_state(context.scene_state)}\n\n"
        f"Latest observation:\n{_latest_observation_summary(context)}\n\n"
        f"Recent history:\n{_history_summary(context.history)}\n\n"
        f"Memory:\n{_memory_summary(context.memory_results)}\n\n"
        f"Screenshot:\n{json.dumps({'available': screenshot.get('available', False), 'path': screenshot.get('path'), 'reason': screenshot.get('reason')}, indent=2)}\n\n"
        "Tool definitions are supplied separately when the provider supports structured tools."
    )


def _serialized_provider_attempts(provider_attempts: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for attempt in provider_attempts:
        serialized.append(
            {
                "provider": getattr(attempt, "provider", ""),
                "model": getattr(attempt, "model", ""),
                "transport": getattr(attempt, "transport", ""),
                "runtime_state": getattr(attempt, "runtime_state", ""),
                "response_type": getattr(attempt, "response_type", ""),
                "parsed_tool_call_count": getattr(attempt, "parsed_tool_call_count", 0),
                "failure_reason": getattr(attempt, "failure_reason", ""),
                "request_metadata": getattr(attempt, "request_metadata", {}),
                "response_metadata": getattr(attempt, "response_metadata", {}),
            }
        )
    return serialized


def _merge_provider_chains(*chains: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for chain in chains:
        for value in chain:
            if not isinstance(value, str) or not value or value in seen:
                continue
            merged.append(value)
            seen.add(value)
    return merged


def _tool_validation_summary(issues: list[ToolCallValidationIssue]) -> str:
    return "; ".join(f"{issue.tool_name}: {issue.reason}" for issue in issues[:4])


def _needs_coordinated_batch(context: DirectorContext) -> bool:
    prompt = context.user_prompt.strip().lower()
    if not prompt:
        return False
    broad_keywords = (
        "scene",
        "room",
        "interior",
        "environment",
        "coherent",
        "cinematic",
        "focal point",
        "composition",
        "lighting",
        "camera",
        "animate",
        "animation",
    )
    if any(keyword in prompt for keyword in broad_keywords):
        return True
    objects = context.scene_state.get("objects", [])
    return isinstance(objects, list) and len(objects) >= 3


def _provider_error_turn(exc: ProviderError, budget_state: BudgetState) -> DirectorTurn:
    runtime_state = getattr(exc, "runtime_state", "provider_transport_failure")
    attempts = _serialized_provider_attempts(getattr(exc, "attempts", []))
    return DirectorTurn(
        status="error",
        message=str(exc),
        narration=str(exc),
        understanding="The Director could not get a usable answer from any configured provider.",
        plan=["surface the provider failure"],
        intended_actions=[],
        expected_outcome="No execution could be prepared.",
        continue_loop=False,
        error=str(exc),
        metadata={
            "bottleneck_tags": ["provider_transport"],
            "budget_state": budget_state.__dict__,
            "runtime_state": runtime_state,
            "runtime_state_detail": str(exc),
            "provider_attempts": attempts,
            "selected_provider": None,
            "response_type": "transport_failure",
            "parsed_tool_call_count": 0,
            "failure_reason": str(exc),
        },
    )


class DirectorLoop:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_default_registry()
        self.registry.discover()

    def _tool_schemas(self, execution_mode: str) -> list[dict[str, Any]]:
        schemas = [_tool_to_schema(self.registry.get(name)) for name in self.registry.list_tools()]
        schemas.extend(_control_tools(execution_mode))
        return schemas

    def _validate_tool_calls(
        self,
        tool_calls: list[ToolCall],
        context: DirectorContext,
    ) -> tuple[list[ToolCall], list[ToolCallValidationIssue]]:
        validated: list[ToolCall] = []
        issues: list[ToolCallValidationIssue] = []
        actionable_families: list[str] = []

        for tool_call in tool_calls[:_BATCH_LIMIT]:
            if not isinstance(tool_call.arguments, dict):
                issues.append(
                    ToolCallValidationIssue(
                        tool_name=tool_call.name,
                        reason="Tool arguments must be a JSON object.",
                    )
                )
                continue

            if tool_call.name == "task.complete":
                summary = str(tool_call.arguments.get("summary", "")).strip()
                if not summary:
                    issues.append(
                        ToolCallValidationIssue(
                            tool_name=tool_call.name,
                            reason="task.complete requires a non-empty summary.",
                        )
                    )
                    continue
                validated.append(ToolCall(name=tool_call.name, arguments={"summary": summary}))
                continue

            if tool_call.name == "task.clarify":
                question = str(tool_call.arguments.get("question", "")).strip()
                reason = str(tool_call.arguments.get("reason", "")).strip()
                if not question or not reason:
                    issues.append(
                        ToolCallValidationIssue(
                            tool_name=tool_call.name,
                            reason="task.clarify requires both question and reason.",
                        )
                    )
                    continue
                validated.append(ToolCall(name=tool_call.name, arguments={"question": question, "reason": reason}))
                continue

            if tool_call.name == "python.execute_blender_snippet":
                if context.execution_mode != "vectra-code":
                    issues.append(
                        ToolCallValidationIssue(
                            tool_name=tool_call.name,
                            reason="Dynamic Blender Python is only available in vectra-code mode.",
                        )
                    )
                    continue
                code = str(tool_call.arguments.get("code", "")).strip()
                if not code:
                    issues.append(
                        ToolCallValidationIssue(
                            tool_name=tool_call.name,
                            reason="python.execute_blender_snippet requires non-empty code.",
                        )
                    )
                    continue
                validated.append(ToolCall(name=tool_call.name, arguments={"code": code}))
                actionable_families.append(_tool_family(tool_call.name))
                continue

            try:
                tool = self.registry.get(tool_call.name)
            except ToolNotFoundError:
                issues.append(
                    ToolCallValidationIssue(
                        tool_name=tool_call.name,
                        reason="The tool is not registered in the live Vectra tool surface.",
                    )
                )
                continue

            try:
                normalized_arguments = tool.validate_params(tool_call.arguments)
            except ToolValidationError as exc:
                issues.append(ToolCallValidationIssue(tool_name=tool_call.name, reason=str(exc)))
                continue

            validated.append(ToolCall(name=tool_call.name, arguments=normalized_arguments))
            actionable_families.append(_tool_family(tool_call.name))

        actionable_count = sum(
            1
            for tool_call in validated
            if tool_call.name not in {"task.complete", "task.clarify"}
        )
        if not issues and actionable_count == 1 and _needs_coordinated_batch(context):
            only_family = actionable_families[0] if actionable_families else "generic"
            if only_family not in {"observe", "code"}:
                issues.append(
                    ToolCallValidationIssue(
                        tool_name=validated[0].name,
                        reason=(
                            "This prompt needs a coordinated batch of 2 to 4 tool calls rather than a single local action."
                        ),
                    )
                )

        return validated, issues

    def _validate_resolved_actions(
        self,
        actions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[ToolCallValidationIssue]]:
        validated_actions: list[dict[str, Any]] = []
        issues: list[ToolCallValidationIssue] = []

        for action in actions:
            tool_name = str(action.get("tool", "")).strip()
            params = action.get("params", {})
            if not tool_name:
                issues.append(ToolCallValidationIssue(tool_name="unknown", reason="Resolved action was missing a tool name."))
                continue
            try:
                tool = self.registry.get(tool_name)
            except ToolNotFoundError:
                issues.append(
                    ToolCallValidationIssue(
                        tool_name=tool_name,
                        reason="Resolved action referenced a tool that is not registered.",
                    )
                )
                continue
            try:
                normalized_params = tool.validate_params(params if isinstance(params, dict) else {})
            except ToolValidationError as exc:
                issues.append(ToolCallValidationIssue(tool_name=tool_name, reason=str(exc)))
                continue
            validated_actions.append({**action, "params": normalized_params})

        return validated_actions, issues

    @staticmethod
    def _tool_validation_retry_prompt(prompt_text: str, issues: list[ToolCallValidationIssue]) -> str:
        return (
            f"{prompt_text}\n\n"
            "Validation correction:\n"
            f"The previous tool batch was not executable because: {_tool_validation_summary(issues)}.\n"
            "Return only supported tools with schema-valid arguments.\n"
            "For broad scene, editing, or animation work, return a coordinated batch of 2 to 4 tool calls unless only one safe action truly exists."
        )

    def _resolve_single_tool_call(
        self,
        tool_call: ToolCall,
        context: DirectorContext,
        *,
        step_index: int,
    ) -> tuple[list[dict[str, Any]], list[AssumptionRecord], dict[str, Any], str | None]:
        resolver = ReferenceResolver(context)
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {
            "chosen_tool": tool_call.name,
            "action_family": _tool_family(tool_call.name),
        }
        args = dict(tool_call.arguments)
        action_id = f"step_{step_index}"

        if tool_call.name == "python.execute_blender_snippet":
            return [], assumptions, metadata, str(args.get("code", "")).strip()

        if tool_call.name in {"task.complete", "task.clarify"}:
            return [], assumptions, metadata, None

        if tool_call.name == "mesh.create_primitive":
            primitive_type = args.get("type", args.get("primitive_type", "cube"))
            resolved_location = resolver.resolve_location(
                "mesh.create_primitive",
                args.get("location"),
                primitive_type=str(primitive_type),
                scale=args.get("scale"),
            )
            assumptions.extend(resolved_location.assumptions)
            metadata["reference_anchor"] = resolved_location.metadata.get("anchor")
            return (
                [
                    {
                        "action_id": action_id,
                        "tool": tool_call.name,
                        "params": _drop_none_values(
                            {
                                "type": primitive_type,
                                "name": args.get("name"),
                                "location": resolved_location.value,
                                "scale": args.get("scale"),
                                "rotation": args.get("rotation"),
                            }
                        ),
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name == "object.transform":
            target_result = resolver.resolve_target(args.get("target", args.get("object_name")))
            assumptions.extend(target_result.assumptions)
            location_result = (
                resolver.resolve_location(
                    "object.transform",
                    args.get("location"),
                    target_name=target_result.value if isinstance(target_result.value, str) else None,
                )
                if args.get("location") is not None
                else None
            )
            if location_result is not None:
                assumptions.extend(location_result.assumptions)
            metadata["reference_anchor"] = target_result.metadata.get("anchor")
            return (
                [
                    {
                        "action_id": action_id,
                        "tool": tool_call.name,
                        "params": _drop_none_values(
                            {
                                "target": target_result.value,
                                "location": location_result.value if location_result is not None else args.get("location"),
                                "delta": args.get("delta"),
                                "rotation": args.get("rotation", args.get("rotation_euler")),
                                "scale": args.get("scale"),
                            }
                        ),
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name in {"object.place_on_surface", "object.place_against", "object.place_relative", "object.align_to"}:
            target_result = resolver.resolve_required_target(args.get("target"), "target")
            reference_result = resolver.resolve_required_target(args.get("reference"), "reference")
            assumptions.extend(target_result.assumptions)
            assumptions.extend(reference_result.assumptions)
            metadata["reference_anchor"] = reference_result.metadata.get("anchor") or target_result.metadata.get("anchor")
            params = dict(args)
            params["target"] = target_result.value
            params["reference"] = reference_result.value
            return (
                [{"action_id": action_id, "tool": tool_call.name, "params": _drop_none_values(params)}],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name in {"object.transform_many", "object.delete_many", "object.distribute", "object.align"}:
            objects_result = resolver.resolve_objects(args.get("targets", args.get("objects")))
            assumptions.extend(objects_result.assumptions)
            metadata["reference_anchor"] = objects_result.metadata.get("anchor")
            params = dict(args)
            params["targets"] = objects_result.value
            return (
                [
                    {
                        "action_id": action_id,
                        "tool": tool_call.name,
                        "params": _drop_none_values(params),
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name in {"object.duplicate", "object.delete", "object.select"}:
            target_result = resolver.resolve_target(args.get("target"))
            assumptions.extend(target_result.assumptions)
            metadata["reference_anchor"] = target_result.metadata.get("anchor")
            params = dict(args)
            params["target"] = target_result.value
            return (
                [{"action_id": action_id, "tool": tool_call.name, "params": _drop_none_values(params)}],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name == "scene.group":
            objects_result = resolver.resolve_objects(args.get("objects"))
            assumptions.extend(objects_result.assumptions)
            metadata["reference_anchor"] = objects_result.metadata.get("anchor")
            return (
                [
                    {
                        "action_id": action_id,
                        "tool": tool_call.name,
                        "params": _drop_none_values({"objects": objects_result.value, "name": args.get("name")}),
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name == "object.parent":
            parent_result = resolver.resolve_target(args.get("parent"))
            child_result = resolver.resolve_objects(args.get("children", args.get("objects")))
            assumptions.extend(parent_result.assumptions)
            assumptions.extend(child_result.assumptions)
            metadata["reference_anchor"] = parent_result.metadata.get("anchor") or child_result.metadata.get("anchor")
            return (
                [
                    {
                        "action_id": action_id,
                        "tool": tool_call.name,
                        "params": _drop_none_values({"parent": parent_result.value, "children": child_result.value}),
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name in {"light.create", "camera.ensure", "light.adjust", "camera.adjust", "material.apply_basic", "object.keyframe"}:
            target_key = "look_at" if tool_call.name == "camera.adjust" else "target"
            target_result = resolver.resolve_target(args.get("target"))
            if tool_call.name != "light.create":
                assumptions.extend(target_result.assumptions)
            resolved_location = None
            if tool_call.name in {"light.create", "camera.ensure", "light.adjust", "camera.adjust"}:
                target_name = target_result.value if isinstance(target_result.value, str) else None
                resolved_location = resolver.resolve_location(tool_call.name, args.get("location"), target_name=target_name)
                assumptions.extend(resolved_location.assumptions)
                metadata["reference_anchor"] = resolved_location.metadata.get("anchor")
            params = dict(args)
            if tool_call.name in {"material.apply_basic", "object.keyframe", "light.adjust", "camera.adjust"}:
                params["target"] = target_result.value
            if resolved_location is not None:
                params["location"] = resolved_location.value
            if tool_call.name == "camera.adjust" and args.get("look_at") is not None:
                look_at_result = resolver.resolve_target(args.get("look_at"))
                assumptions.extend(look_at_result.assumptions)
                params[target_key] = look_at_result.value
            return (
                [{"action_id": action_id, "tool": tool_call.name, "params": _drop_none_values(params)}],
                assumptions,
                metadata,
                None,
            )

        return (
            [{"action_id": action_id, "tool": tool_call.name, "params": _drop_none_values(args)}],
            assumptions,
            metadata,
            None,
        )

    def _resolve_tool_calls(
        self,
        tool_calls: list[ToolCall],
        context: DirectorContext,
    ) -> tuple[list[dict[str, Any]], list[AssumptionRecord], dict[str, Any], str | None]:
        actions: list[dict[str, Any]] = []
        assumptions: list[AssumptionRecord] = []
        reference_anchors: list[str | None] = []
        action_families: list[str] = []
        code: str | None = None

        for step_index, tool_call in enumerate(tool_calls[:_BATCH_LIMIT], start=1):
            resolved_actions, resolved_assumptions, metadata, resolved_code = self._resolve_single_tool_call(
                tool_call,
                context,
                step_index=step_index,
            )
            actions.extend(resolved_actions)
            assumptions.extend(resolved_assumptions)
            reference_anchors.append(metadata.get("reference_anchor"))
            if isinstance(metadata.get("action_family"), str):
                action_families.append(metadata["action_family"])
            if resolved_code:
                code = resolved_code

        return (
            actions,
            assumptions,
            {
                "reference_anchors": reference_anchors,
                "action_families": action_families,
                "batch_size": len(actions),
            },
            code,
        )

    @staticmethod
    def _plan_lines(provider_result: ProviderResult, tool_calls: list[ToolCall]) -> list[str]:
        if tool_calls:
            return [f"execute a batch using {', '.join(tool.name for tool in tool_calls[:_BATCH_LIMIT])}"]
        if provider_result.assistant_text:
            return [provider_result.assistant_text]
        return ["decide the next best scene step"]

    @staticmethod
    def _intended_actions(tool_calls: list[ToolCall]) -> list[str]:
        return [f"{tool.name}({tool.arguments})" for tool in tool_calls[:_BATCH_LIMIT]]

    def step(self, context: DirectorContext) -> DirectorTurn:
        controller_decision = call_controller(context.user_prompt, context.scene_state)
        budget_state = context.budget_state or _derive_budget_state(context, controller_decision)
        tools = self._tool_schemas(context.execution_mode)
        prompt_text = _build_director_prompt(context, controller_decision, budget_state)

        try:
            provider_result = call_director(
                prompt_text=prompt_text,
                tools=tools,
                allow_complete=context.iteration > 1,
            )
        except ProviderError as exc:
            return _provider_error_turn(exc, budget_state)

        provider_attempts = list(provider_result.attempts)
        provider_chain = _merge_provider_chains(
            provider_result.provider_chain or [f"{provider_result.provider}:{provider_result.model}"]
        )
        tool_calls = provider_result.tool_calls[:_BATCH_LIMIT]
        validation_issues: list[ToolCallValidationIssue] = []
        actions: list[dict[str, Any]] = []
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {}
        code: str | None = None
        validation_retry_used = False

        for _ in range(2):
            if not tool_calls or tool_calls[0].name in {"task.complete", "task.clarify"}:
                break

            validated_tool_calls, validation_issues = self._validate_tool_calls(tool_calls, context)
            if validation_issues:
                if validation_retry_used:
                    break
                validation_retry_used = True
                log_structured(
                    _LOGGER,
                    "tool_batch_validation_failed",
                    {
                        "provider": provider_result.provider,
                        "model": provider_result.model,
                        "transport": provider_result.transport,
                        "issue_count": len(validation_issues),
                        "issue_summary": _tool_validation_summary(validation_issues),
                    },
                    level="warning",
                )
                try:
                    provider_result = call_director(
                        prompt_text=self._tool_validation_retry_prompt(prompt_text, validation_issues),
                        tools=tools,
                        allow_complete=context.iteration > 1,
                    )
                except ProviderError as exc:
                    return _provider_error_turn(exc, budget_state)
                provider_attempts.extend(provider_result.attempts)
                provider_chain = _merge_provider_chains(
                    provider_chain,
                    provider_result.provider_chain or [f"{provider_result.provider}:{provider_result.model}"],
                )
                tool_calls = provider_result.tool_calls[:_BATCH_LIMIT]
                validation_issues = []
                continue

            actions, assumptions, metadata, code = self._resolve_tool_calls(validated_tool_calls, context)
            validated_actions, action_issues = self._validate_resolved_actions(actions)
            if action_issues:
                validation_issues = action_issues
                if validation_retry_used:
                    break
                validation_retry_used = True
                log_structured(
                    _LOGGER,
                    "resolved_action_validation_failed",
                    {
                        "provider": provider_result.provider,
                        "model": provider_result.model,
                        "transport": provider_result.transport,
                        "issue_count": len(validation_issues),
                        "issue_summary": _tool_validation_summary(validation_issues),
                    },
                    level="warning",
                )
                try:
                    provider_result = call_director(
                        prompt_text=self._tool_validation_retry_prompt(prompt_text, validation_issues),
                        tools=tools,
                        allow_complete=context.iteration > 1,
                    )
                except ProviderError as exc:
                    return _provider_error_turn(exc, budget_state)
                provider_attempts.extend(provider_result.attempts)
                provider_chain = _merge_provider_chains(
                    provider_chain,
                    provider_result.provider_chain or [f"{provider_result.provider}:{provider_result.model}"],
                )
                tool_calls = provider_result.tool_calls[:_BATCH_LIMIT]
                actions = []
                assumptions = []
                metadata = {}
                code = None
                validation_issues = []
                continue

            tool_calls = validated_tool_calls
            actions = validated_actions
            validation_issues = []
            break

        if tool_calls and tool_calls[0].name == "task.complete":
            summary = str(tool_calls[0].arguments.get("summary", "")).strip() or provider_result.assistant_text or "Task complete."
            return DirectorTurn(
                status="complete",
                message=summary,
                narration=provider_result.assistant_text or summary,
                understanding="The Director marked the task complete.",
                plan=["finish the task"],
                intended_actions=[],
                expected_outcome=summary,
                continue_loop=False,
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={
                    "provider_result": provider_result.raw_response,
                    "provider_chain": provider_chain,
                    "provider_attempts": _serialized_provider_attempts(provider_attempts),
                    "budget_state": budget_state.__dict__,
                    "bottleneck_tags": [],
                    "runtime_state": provider_result.runtime_state,
                    "runtime_state_detail": summary,
                    "selected_provider": provider_result.provider,
                    "response_type": provider_result.response_type,
                    "parsed_tool_call_count": len(provider_result.tool_calls),
                    "failure_reason": "",
                    "validation_retry_used": validation_retry_used,
                },
            )

        if tool_calls and tool_calls[0].name == "task.clarify":
            question = str(tool_calls[0].arguments.get("question", "")).strip() or "I need clarification before continuing."
            reason = str(tool_calls[0].arguments.get("reason", "")).strip()
            return DirectorTurn(
                status="clarify",
                message=question,
                narration=provider_result.assistant_text or question,
                understanding=reason or "The Director is blocked on missing capability or unsafe execution.",
                plan=["ask for clarification"],
                intended_actions=[],
                expected_outcome="Await clarification before continuing.",
                continue_loop=False,
                question=question,
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={
                    "provider_result": provider_result.raw_response,
                    "provider_chain": provider_chain,
                    "provider_attempts": _serialized_provider_attempts(provider_attempts),
                    "budget_state": budget_state.__dict__,
                    "bottleneck_tags": ["tool_gap"] if reason else [],
                    "runtime_state": provider_result.runtime_state,
                    "runtime_state_detail": question,
                    "selected_provider": provider_result.provider,
                    "response_type": provider_result.response_type,
                    "parsed_tool_call_count": len(provider_result.tool_calls),
                    "failure_reason": reason,
                    "validation_retry_used": validation_retry_used,
                },
            )

        if not tool_calls:
            return DirectorTurn(
                status="error",
                message="The Director did not produce a usable action batch or completion signal.",
                narration="The Director stalled without choosing a next action.",
                understanding="The provider response was not actionable.",
                plan=["surface an execution error"],
                intended_actions=[],
                expected_outcome="No execution could be prepared.",
                continue_loop=False,
                error="No usable tool call returned",
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={
                    "provider_result": provider_result.raw_response,
                    "provider_chain": provider_chain,
                    "provider_attempts": _serialized_provider_attempts(provider_attempts),
                    "budget_state": budget_state.__dict__,
                    "bottleneck_tags": ["provider_transport"],
                    "runtime_state": "no_action_response",
                    "runtime_state_detail": "The provider returned no usable tool calls.",
                    "selected_provider": provider_result.provider,
                    "response_type": provider_result.response_type,
                    "parsed_tool_call_count": 0,
                    "failure_reason": provider_result.failure_reason or "No usable tool call returned",
                    "validation_retry_used": validation_retry_used,
                },
            )

        if validation_issues:
            failure_reason = f"The provider produced unsupported or invalid tool calls: {_tool_validation_summary(validation_issues)}"
            return DirectorTurn(
                status="error",
                message=failure_reason,
                narration=failure_reason,
                understanding="The Director returned tool calls that did not match the live tool surface or schema.",
                plan=["request a stricter executable batch"],
                intended_actions=self._intended_actions(tool_calls),
                expected_outcome="No execution could be prepared.",
                continue_loop=False,
                error=failure_reason,
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={
                    "provider_result": provider_result.raw_response,
                    "provider_chain": provider_chain,
                    "provider_attempts": _serialized_provider_attempts(provider_attempts),
                    "budget_state": budget_state.__dict__,
                    "bottleneck_tags": ["tool_grounding"],
                    "runtime_state": "tool_validation_failure",
                    "runtime_state_detail": failure_reason,
                    "selected_provider": provider_result.provider,
                    "response_type": provider_result.response_type,
                    "parsed_tool_call_count": len(provider_result.tool_calls),
                    "failure_reason": failure_reason,
                    "tool_validation_issues": [
                        {"tool_name": issue.tool_name, "reason": issue.reason}
                        for issue in validation_issues
                    ],
                    "validation_retry_used": validation_retry_used,
                },
            )

        runtime_state = provider_result.runtime_state
        runtime_state_detail = (
            f"Valid action batch ready via fallback provider {provider_result.provider}."
            if runtime_state == "fallback_provider_invoked"
            else f"Valid action batch ready from {provider_result.provider}."
        )
        if not actions and not code:
            failure_reason = "The provider response was structured but did not resolve into executable actions."
            log_structured(
                _LOGGER,
                "non_actionable_response_rejected",
                {
                    "provider": provider_result.provider,
                    "model": provider_result.model,
                    "transport": provider_result.transport,
                    "runtime_state": "no_action_response",
                    "failure_reason": failure_reason,
                },
                level="error",
            )
            return DirectorTurn(
                status="error",
                message=failure_reason,
                narration=failure_reason,
                understanding="The Director returned structured output that could not be turned into executable work.",
                plan=["surface a structured failure"],
                intended_actions=self._intended_actions(tool_calls),
                expected_outcome="No execution could be prepared.",
                continue_loop=False,
                error=failure_reason,
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={
                    "provider_result": provider_result.raw_response,
                    "provider_chain": provider_chain,
                    "provider_attempts": _serialized_provider_attempts(provider_attempts),
                    "budget_state": budget_state.__dict__,
                    "bottleneck_tags": ["provider_transport"],
                    "runtime_state": "no_action_response",
                    "runtime_state_detail": failure_reason,
                    "selected_provider": provider_result.provider,
                    "response_type": provider_result.response_type,
                    "parsed_tool_call_count": len(provider_result.tool_calls),
                    "failure_reason": failure_reason,
                    "validation_retry_used": validation_retry_used,
                },
            )
        metadata = dict(metadata)
        metadata.update(
            {
                "provider": provider_result.provider,
                "model": provider_result.model,
                "transport": provider_result.transport,
                "provider_result": provider_result.raw_response,
                "provider_chain": provider_chain,
                "provider_attempts": _serialized_provider_attempts(provider_attempts),
                "code": code,
                "assumptions": [
                    {"key": item.key, "value": item.value, "reason": item.reason}
                    for item in assumptions
                ],
                "actions": actions,
                "budget_state": budget_state.__dict__,
                "bottleneck_tags": [],
                "observation_summary": _latest_observation_summary(context),
                "completion_mode_active": budget_state.completion_mode_active,
                "turn_index": context.iteration,
                "runtime_state": runtime_state,
                "runtime_state_detail": runtime_state_detail,
                "selected_provider": provider_result.provider,
                "response_type": provider_result.response_type,
                "parsed_tool_call_count": len(provider_result.tool_calls),
                "failure_reason": "",
                "validation_retry_used": validation_retry_used,
            }
        )
        log_structured(
            _LOGGER,
            "action_batch_accepted",
            {
                "provider": provider_result.provider,
                "model": provider_result.model,
                "transport": provider_result.transport,
                "runtime_state": runtime_state,
                "response_type": provider_result.response_type,
                "parsed_tool_call_count": len(provider_result.tool_calls),
                "selected_action_count": len(actions),
                "selected_action_families": metadata.get("action_families", []),
            },
        )
        return DirectorTurn(
            status="ok",
            message=provider_result.assistant_text or "Prepared the next director step.",
            narration=provider_result.assistant_text or "Prepared the next director step.",
            understanding="The Director selected the next bounded batch using scene context, observations, and the turn budget.",
            plan=self._plan_lines(provider_result, tool_calls),
            intended_actions=self._intended_actions(tool_calls),
            expected_outcome=provider_result.assistant_text or "The scene should move closer to the request.",
            continue_loop=True,
            assumptions=assumptions,
            tool_calls=tool_calls,
            code=code,
            provider=provider_result.provider,
            model=provider_result.model,
            metadata=metadata,
        )
