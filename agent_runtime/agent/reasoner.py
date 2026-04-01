from __future__ import annotations

import json
from typing import Any

from agent_runtime.planner import plan as legacy_plan

from .models import AgentContext, ExecutionMode, ReasoningStep

_CUBE_NAMES = ("cube", "vectracube", "stair")


def _mesh_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    objects = scene_state.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict) and obj.get("type") == "MESH"]


def _cube_like_objects(scene_state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for obj in _mesh_objects(scene_state):
        name = str(obj.get("name", "")).lower()
        if any(token in name for token in _CUBE_NAMES):
            candidates.append(obj)
            continue
        dimensions = obj.get("dimensions")
        if isinstance(dimensions, list) and len(dimensions) == 3:
            width, depth, height = (float(value) for value in dimensions)
            if abs(width - depth) < 0.001 and abs(depth - height) < 0.001:
                candidates.append(obj)
    return candidates


def _location(obj: dict[str, Any]) -> list[float]:
    raw_value = obj.get("location")
    if isinstance(raw_value, list) and len(raw_value) == 3:
        return [float(component) for component in raw_value]
    return [0.0, 0.0, 0.0]


def _dimensions(obj: dict[str, Any]) -> list[float]:
    raw_value = obj.get("dimensions")
    if isinstance(raw_value, list) and len(raw_value) == 3:
        return [float(component) for component in raw_value]
    return [2.0, 2.0, 2.0]


def _name(obj: dict[str, Any], fallback: str) -> str:
    raw_value = obj.get("name")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value
    return fallback


def _make_reasoning(
    *,
    narration: str,
    understanding: str,
    plan: list[str],
    intended_actions: list[str],
    expected_outcome: str,
    preferred_execution_mode: ExecutionMode,
    continue_loop: bool,
    status: str = "ok",
    question: str | None = None,
    uncertainty_notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReasoningStep:
    return ReasoningStep(
        status=status,  # type: ignore[arg-type]
        narration=narration,
        understanding=understanding,
        plan=plan,
        intended_actions=intended_actions,
        expected_outcome=expected_outcome,
        preferred_execution_mode=preferred_execution_mode,
        continue_loop=continue_loop,
        question=question,
        uncertainty_notes=uncertainty_notes or [],
        metadata=metadata or {},
    )


def _two_cubes_reasoning(context: AgentContext) -> ReasoningStep:
    cubes = _cube_like_objects(context.scene_state)
    if len(cubes) >= 2:
        return _make_reasoning(
            narration="Two cubes are already in the scene, so I can stop here.",
            understanding="The task is to ensure there are two cubes available.",
            plan=["Confirm the target object count.", "Stop once two cubes exist."],
            intended_actions=[],
            expected_outcome="The scene contains at least two cubes.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="complete",
        )

    index = len(cubes) + 1
    name = f"VectraCube{index}"
    location = [0.0, 0.0, 0.0] if index == 1 else [1.0, 0.0, 0.0]
    return _make_reasoning(
        narration=f"Creating cube {index} so the scene reaches two cubes.",
        understanding="The user wants exactly two cubes in the scene.",
        plan=["Check how many cubes already exist.", "Create the missing cube.", "Repeat until two cubes exist."],
        intended_actions=[
            f"create a cube named {name} at ({location[0]:.1f}, {location[1]:.1f}, {location[2]:.1f})"
        ],
        expected_outcome="The scene will contain one more cube after this step.",
        preferred_execution_mode=context.execution_mode,
        continue_loop=True,
    )


def _move_apart_reasoning(context: AgentContext) -> ReasoningStep:
    cubes = _cube_like_objects(context.scene_state)
    if len(cubes) < 2:
        return _make_reasoning(
            narration="I need two cubes before I can separate them.",
            understanding="The user wants the existing cubes spaced apart.",
            plan=["Find two cube-like objects.", "Move them to a clearly separated layout."],
            intended_actions=[],
            expected_outcome="Two cubes become separated horizontally.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="clarify",
            question="I need two cubes in the scene before I can move them apart.",
            uncertainty_notes=["No safe pair of cubes was available to separate."],
        )

    first, second = sorted(cubes, key=lambda obj: _name(obj, "Cube"))
    first_location = _location(first)
    second_location = _location(second)
    if abs(second_location[0] - first_location[0]) >= 2.5:
        return _make_reasoning(
            narration="The cubes already have clear horizontal spacing.",
            understanding="The user wants two cubes separated in the scene.",
            plan=["Measure the current spacing.", "Stop if they are already clearly apart."],
            intended_actions=[],
            expected_outcome="The cubes remain clearly separated.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="complete",
        )

    return _make_reasoning(
        narration="Moving the cubes apart horizontally for clearer spacing.",
        understanding="The user wants two cubes moved away from each other.",
        plan=["Keep one cube to the left.", "Move the other cube to the right."],
        intended_actions=[
            f"move object {_name(first, 'Cube')} to (-1.5, {first_location[1]:.1f}, {first_location[2]:.1f})",
            f"move object {_name(second, 'Cube')} to (1.5, {second_location[1]:.1f}, {second_location[2]:.1f})",
        ],
        expected_outcome="The cubes end this step with clear horizontal spacing.",
        preferred_execution_mode=context.execution_mode,
        continue_loop=False,
    )


def _staircase_reasoning(context: AgentContext) -> ReasoningStep:
    cubes = sorted(_cube_like_objects(context.scene_state), key=lambda obj: _name(obj, "Cube"))
    target_count = 5
    if len(cubes) >= target_count:
        return _make_reasoning(
            narration="The staircase already has enough steps, so I’m stopping.",
            understanding="The user wants a staircase built from cubes.",
            plan=["Count the staircase cubes.", "Stop when the target step count is reached."],
            intended_actions=[],
            expected_outcome="The scene contains a staircase of cubes.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="complete",
        )

    step_index = len(cubes)
    name = f"VectraStair{step_index + 1}"
    location = [step_index * 1.25, 0.0, step_index * 1.0]
    return _make_reasoning(
        narration=f"Adding staircase step {step_index + 1}.",
        understanding="The user wants a staircase pattern made from cubes.",
        plan=["Count the existing steps.", "Create the next cube forward and upward.", "Repeat until the staircase is complete."],
        intended_actions=[
            f"create a cube named {name} at ({location[0]:.2f}, {location[1]:.1f}, {location[2]:.2f})"
        ],
        expected_outcome="The staircase gains one additional step.",
        preferred_execution_mode=context.execution_mode,
        continue_loop=True,
    )


def _stack_reasoning(context: AgentContext) -> ReasoningStep:
    cubes = sorted(_cube_like_objects(context.scene_state), key=lambda obj: _name(obj, "Cube"))
    if not cubes:
        return _make_reasoning(
            narration="Creating the base cube before stacking anything on top.",
            understanding="The task requires one cube to act as a base and another cube above it.",
            plan=["Create a base cube.", "Create or move another cube onto it."],
            intended_actions=["create a cube named VectraBaseCube at (0.0, 0.0, 0.0)"],
            expected_outcome="The scene will contain a base cube for stacking.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=True,
        )

    if len(cubes) == 1:
        base = cubes[0]
        base_location = _location(base)
        base_dimensions = _dimensions(base)
        top_location = [
            base_location[0],
            base_location[1],
            base_location[2] + max(base_dimensions[2], 2.0),
        ]
        return _make_reasoning(
            narration="Creating the second cube directly above the base cube.",
            understanding="The task is to place one cube on top of another.",
            plan=["Use the existing cube as the base.", "Create the top cube at the base cube's top surface."],
            intended_actions=[
                f"create a cube named VectraTopCube at ({top_location[0]:.1f}, {top_location[1]:.1f}, {top_location[2]:.1f})"
            ],
            expected_outcome="The scene will contain a second cube positioned above the base.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=True,
        )

    base = cubes[0]
    top = cubes[1]
    base_location = _location(base)
    top_location = _location(top)
    base_dimensions = _dimensions(base)
    target_location = [
        base_location[0],
        base_location[1],
        base_location[2] + max(base_dimensions[2], 2.0),
    ]
    if all(abs(top_location[index] - target_location[index]) < 0.05 for index in range(3)):
        return _make_reasoning(
            narration="The top cube is already stacked correctly.",
            understanding="The goal is to have one cube resting on top of another.",
            plan=["Check the vertical alignment of the second cube.", "Stop if it is already stacked correctly."],
            intended_actions=[],
            expected_outcome="The cubes stay stacked vertically.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="complete",
        )

    return _make_reasoning(
        narration="Moving the second cube so it sits on top of the first one.",
        understanding="The user wants a clean vertical stack of two cubes.",
        plan=["Use the first cube as the base.", "Move the second cube to the base cube's top surface."],
        intended_actions=[
            f"move object {_name(top, 'Cube')} to ({target_location[0]:.1f}, {target_location[1]:.1f}, {target_location[2]:.1f})"
        ],
        expected_outcome="The second cube ends the step stacked on top of the first.",
        preferred_execution_mode=context.execution_mode,
        continue_loop=False,
    )


def _matches(prompt: str, *patterns: str) -> bool:
    lowered = prompt.strip().lower()
    return any(pattern in lowered for pattern in patterns)


def _fallback_legacy_reasoning(context: AgentContext) -> ReasoningStep:
    planner_result = legacy_plan(context.user_prompt, context.scene_state)
    if planner_result.status != "ok" or not planner_result.actions:
        return _make_reasoning(
            narration=planner_result.message,
            understanding="The agent could not ground a safe plan from the current context.",
            plan=["Try the structured tool planner as a fallback."],
            intended_actions=[],
            expected_outcome="Either a valid fallback plan is found, or the agent stops safely.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=False,
            status="error",
            uncertainty_notes=[planner_result.message],
        )

    intended_actions: list[str] = []
    for action in planner_result.actions:
        tool = action.get("tool")
        params = action.get("params", {})
        if tool == "mesh.create_primitive":
            primitive_type = params.get("primitive_type", "cube")
            name = params.get("name", "VectraObject")
            location = params.get("location", [0.0, 0.0, 0.0])
            intended_actions.append(
                f"create a {primitive_type} named {name} at ({float(location[0]):.1f}, {float(location[1]):.1f}, {float(location[2]):.1f})"
            )
        elif tool == "object.transform":
            object_name = params.get("object_name", "Object")
            location = params.get("location")
            if isinstance(location, list) and len(location) == 3:
                intended_actions.append(
                    f"move object {object_name} to ({float(location[0]):.1f}, {float(location[1]):.1f}, {float(location[2]):.1f})"
                )

    return _make_reasoning(
        narration="Applying a tool-grounded fallback plan.",
        understanding="The prompt did not match an agent-specific task pattern, so the tool planner is handling it.",
        plan=["Run the existing structured planner.", "Execute the resulting tool actions."],
        intended_actions=intended_actions,
        expected_outcome="The fallback tool plan is applied once.",
        preferred_execution_mode=context.execution_mode,
        continue_loop=False,
        metadata={"fallback_actions": planner_result.actions},
    )


def reason_step(context: AgentContext) -> ReasoningStep:
    prompt = context.user_prompt.strip().lower()
    if _matches(prompt, "two cubes", "2 cubes"):
        return _two_cubes_reasoning(context)
    if "staircase" in prompt and "cube" in prompt:
        return _staircase_reasoning(context)
    if "on top of" in prompt and "cube" in prompt:
        return _stack_reasoning(context)
    if "apart" in prompt:
        return _move_apart_reasoning(context)
    return _fallback_legacy_reasoning(context)
