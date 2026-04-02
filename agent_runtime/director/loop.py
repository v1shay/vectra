from __future__ import annotations

import json
from typing import Any

from vectra.tools.base import BaseTool
from vectra.tools.registry import ToolRegistry, get_default_registry

from .models import (
    AssumptionRecord,
    ControllerDecision,
    DirectorContext,
    DirectorTurn,
    ProviderResult,
    ToolCall,
)
from .providers import ProviderError, call_controller, call_director
from .resolver import ReferenceResolver


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


def _compact_scene_state(scene_state: dict[str, Any]) -> str:
    objects = scene_state.get("objects", [])
    compact_objects: list[str] = []
    if isinstance(objects, list):
        for obj in objects[:30]:
            if not isinstance(obj, dict):
                continue
            compact_objects.append(
                "name={name} type={type} active={active} selected={selected} "
                "location={location} dimensions={dimensions}".format(
                    name=obj.get("name"),
                    type=obj.get("type"),
                    active=obj.get("active", False),
                    selected=obj.get("selected", False),
                    location=obj.get("location"),
                    dimensions=obj.get("dimensions"),
                )
            )
    return (
        f"Active object: {scene_state.get('active_object')}\n"
        f"Selected objects: {scene_state.get('selected_objects', [])}\n"
        f"Objects:\n- " + "\n- ".join(compact_objects)
        if compact_objects
        else (
            f"Active object: {scene_state.get('active_object')}\n"
            f"Selected objects: {scene_state.get('selected_objects', [])}\n"
            "Objects: none"
        )
    )


def _history_summary(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No prior execution history."
    snippets: list[str] = []
    for entry in history[-6:]:
        if not isinstance(entry, dict):
            continue
        snippets.append(
            f"[{entry.get('role', 'unknown')}] {entry.get('summary', '')}"
        )
    return "\n".join(snippets) if snippets else "No prior execution history."


def _memory_summary(memory_results: list[dict[str, Any]]) -> str:
    if not memory_results:
        return "No relevant memory."
    return "\n".join(
        str(record.get("summary", "")) for record in memory_results[:5]
    )


def _observation_summary(context: DirectorContext) -> str:
    if context.latest_observation is None:
        return "No fresh observation summary yet."
    return context.latest_observation.summary


def _build_director_prompt(context: DirectorContext, decision: ControllerDecision, tools: list[dict[str, Any]]) -> str:
    screenshot = context.screenshot or {}
    return (
        f"User prompt:\n{context.user_prompt}\n\n"
        f"Controller decision:\n{json.dumps(decision.raw or {'task_type': decision.task_type, 'needs_scene_context': decision.needs_scene_context, 'needs_visual_feedback': decision.needs_visual_feedback, 'complexity': decision.complexity}, indent=2)}\n\n"
        f"Scene state:\n{_compact_scene_state(context.scene_state)}\n\n"
        f"Latest observation:\n{_observation_summary(context)}\n\n"
        f"Recent history:\n{_history_summary(context.history)}\n\n"
        f"Memory:\n{_memory_summary(context.memory_results)}\n\n"
        f"Screenshot:\n{json.dumps({'available': screenshot.get('available', False), 'path': screenshot.get('path'), 'reason': screenshot.get('reason')}, indent=2)}\n\n"
        "Available tools:\n"
        + json.dumps(tools, indent=2)
    )


class DirectorLoop:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_default_registry()
        self.registry.discover()

    def _tool_schemas(self, execution_mode: str) -> list[dict[str, Any]]:
        schemas = [_tool_to_schema(self.registry.get(name)) for name in self.registry.list_tools()]
        schemas.extend(_control_tools(execution_mode))
        return schemas

    def _resolve_tool_call(self, tool_call: ToolCall, context: DirectorContext) -> tuple[list[dict[str, Any]], list[AssumptionRecord], dict[str, Any], str | None]:
        resolver = ReferenceResolver(context)
        assumptions: list[AssumptionRecord] = []
        metadata: dict[str, Any] = {"chosen_tool": tool_call.name}
        args = dict(tool_call.arguments)

        if tool_call.name == "python.execute_blender_snippet":
            return [], assumptions, metadata, str(args.get("code", "")).strip()

        if tool_call.name in {"task.complete", "task.clarify"}:
            return [], assumptions, metadata, None

        if tool_call.name == "mesh.create_primitive":
            primitive_type = args.get("type", args.get("primitive_type", "cube"))
            resolved_location = resolver.resolve_location("mesh.create_primitive", args.get("location"))
            assumptions.extend(resolved_location.assumptions)
            metadata["reference_anchor"] = resolved_location.metadata.get("anchor")
            return (
                [
                    {
                        "action_id": f"director_{context.iteration}_create",
                        "tool": tool_call.name,
                        "params": {
                            "type": primitive_type,
                            "name": args.get("name"),
                            "location": resolved_location.value,
                            "scale": args.get("scale"),
                            "rotation": args.get("rotation"),
                        },
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name == "object.transform":
            target_result = resolver.resolve_target(args.get("target", args.get("object_name")))
            assumptions.extend(target_result.assumptions)
            location_result = resolver.resolve_location(
                "object.transform",
                args.get("location"),
                target_name=target_result.value if isinstance(target_result.value, str) else None,
            ) if "location" in args and args.get("location") is not None else None
            if location_result is not None:
                assumptions.extend(location_result.assumptions)
            metadata["reference_anchor"] = target_result.metadata.get("anchor")
            params = {
                "target": target_result.value,
                "location": location_result.value if location_result is not None else args.get("location"),
                "delta": args.get("delta"),
                "rotation": args.get("rotation", args.get("rotation_euler")),
                "scale": args.get("scale"),
            }
            return (
                [
                    {
                        "action_id": f"director_{context.iteration}_transform",
                        "tool": tool_call.name,
                        "params": params,
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
                [
                    {
                        "action_id": f"director_{context.iteration}_{tool_call.name.replace('.', '_')}",
                        "tool": tool_call.name,
                        "params": params,
                    }
                ],
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
                        "action_id": f"director_{context.iteration}_group",
                        "tool": tool_call.name,
                        "params": {
                            "objects": objects_result.value,
                            "name": args.get("name"),
                        },
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        if tool_call.name in {"light.create", "camera.ensure", "material.apply_basic"}:
            target_result = resolver.resolve_target(args.get("target"))
            assumptions.extend(target_result.assumptions)
            resolved_location = None
            if tool_call.name in {"light.create", "camera.ensure"}:
                resolved_location = resolver.resolve_location(
                    tool_call.name,
                    args.get("location"),
                    target_name=target_result.value if isinstance(target_result.value, str) else None,
                )
                assumptions.extend(resolved_location.assumptions)
                metadata["reference_anchor"] = resolved_location.metadata.get("anchor")
            params = dict(args)
            params["target"] = target_result.value
            if resolved_location is not None:
                params["location"] = resolved_location.value
            return (
                [
                    {
                        "action_id": f"director_{context.iteration}_{tool_call.name.replace('.', '_')}",
                        "tool": tool_call.name,
                        "params": params,
                    }
                ],
                assumptions,
                metadata,
                None,
            )

        return (
            [
                {
                    "action_id": f"director_{context.iteration}_{tool_call.name.replace('.', '_')}",
                    "tool": tool_call.name,
                    "params": args,
                }
            ],
            assumptions,
            metadata,
            None,
        )

    @staticmethod
    def _plan_lines(provider_result: ProviderResult, tool_calls: list[ToolCall]) -> list[str]:
        if tool_calls:
            return [f"use {tool_calls[0].name} as the next step"]
        if provider_result.assistant_text:
            return [provider_result.assistant_text]
        return ["decide the next best scene step"]

    @staticmethod
    def _intended_actions(tool_calls: list[ToolCall]) -> list[str]:
        return [f"{tool.name}({tool.arguments})" for tool in tool_calls]

    def step(self, context: DirectorContext) -> DirectorTurn:
        controller_decision = call_controller(context.user_prompt, context.scene_state)
        tools = self._tool_schemas(context.execution_mode)
        prompt_text = _build_director_prompt(context, controller_decision, tools)
        try:
            provider_result = call_director(prompt_text=prompt_text, tools=tools)
        except ProviderError as exc:
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
                metadata={},
            )

        tool_calls = provider_result.tool_calls
        if not tool_calls and provider_result.assistant_text:
            return DirectorTurn(
                status="complete",
                message=provider_result.assistant_text,
                narration=provider_result.assistant_text,
                understanding="The Director concluded that no further action is required.",
                plan=["finish the task"],
                intended_actions=[],
                expected_outcome=provider_result.assistant_text,
                continue_loop=False,
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={"provider_result": provider_result.raw_response},
            )

        if not tool_calls:
            return DirectorTurn(
                status="error",
                message="The Director did not produce a usable tool call or completion signal.",
                narration="The Director stalled without choosing a next action.",
                understanding="The provider response was not actionable.",
                plan=["surface an execution error"],
                intended_actions=[],
                expected_outcome="No execution could be prepared.",
                continue_loop=False,
                error="No usable tool call returned",
                provider=provider_result.provider,
                model=provider_result.model,
                metadata={"provider_result": provider_result.raw_response},
            )

        first_tool = tool_calls[0]
        if first_tool.name == "task.complete":
            summary = str(first_tool.arguments.get("summary", "")).strip() or provider_result.assistant_text or "Task complete."
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
                metadata={"provider_result": provider_result.raw_response},
            )
        if first_tool.name == "task.clarify":
            question = str(first_tool.arguments.get("question", "")).strip() or "I need clarification before continuing."
            reason = str(first_tool.arguments.get("reason", "")).strip()
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
                metadata={"provider_result": provider_result.raw_response},
            )

        actions, assumptions, metadata, code = self._resolve_tool_call(first_tool, context)
        metadata = dict(metadata)
        metadata.update(
            {
                "provider": provider_result.provider,
                "model": provider_result.model,
                "provider_result": provider_result.raw_response,
                "code": code,
                "assumptions": [
                    {"key": item.key, "value": item.value, "reason": item.reason}
                    for item in assumptions
                ],
            }
        )
        return DirectorTurn(
            status="ok",
            message=provider_result.assistant_text or "Prepared the next director step.",
            narration=provider_result.assistant_text or "Prepared the next director step.",
            understanding=(
                "The Director selected the next bounded action using scene context and recent observations."
            ),
            plan=self._plan_lines(provider_result, tool_calls),
            intended_actions=self._intended_actions(tool_calls),
            expected_outcome=provider_result.assistant_text or "The scene should move one step closer to the request.",
            continue_loop=True,
            assumptions=assumptions,
            tool_calls=[first_tool],
            code=code,
            provider=provider_result.provider,
            model=provider_result.model,
            metadata=metadata | {"actions": actions},
        )
