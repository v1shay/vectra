from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_runtime.agent.models import AgentContext, ReasoningStep

from .graphs import (
    Dependency,
    IntentGoal,
    IntentGraph,
    IntentTask,
    SemanticGraph,
    WorldGraph,
    semantic_graph_for_maintenance_bay,
    world_graph_from_scene_state,
)
from .validation import ValidationResult, validate_maintenance_bay_graph


BENCHMARK_NAME = "maintenance_bay"


@dataclass(frozen=True)
class PlanStep:
    id: str
    skill_name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    expected_graph_delta: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlanExecutionReport:
    executed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    validation_results: ValidationResult | None = None
    remaining_obligations: list[str] = field(default_factory=list)


def is_maintenance_bay_prompt(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    markers = ("maintenance", "bay", "catwalk", "workstation")
    return "maintenance" in normalized and any(marker in normalized for marker in markers[1:])


class MaintenanceBayPlanner:
    """HTN-style vertical slice for Vectra's first graph-backed benchmark."""

    def build_world_graph(self, scene_state: dict[str, Any]) -> WorldGraph:
        return world_graph_from_scene_state(scene_state)

    def build_semantic_graph(self, world_graph: WorldGraph) -> SemanticGraph:
        return semantic_graph_for_maintenance_bay(world_graph)

    def build_intent_graph(self, prompt: str) -> IntentGraph:
        tasks = [
            IntentTask("create_floor", "Create a Z-up horizontal floor slab.", "horizontal floor", expected_roles=["floor"]),
            IntentTask(
                "create_catwalk",
                "Create a raised catwalk above the bay.",
                "raised catwalk",
                dependencies=["create_floor"],
                expected_roles=["catwalk"],
            ),
            IntentTask(
                "create_workstations",
                "Create three workstations beneath the catwalk.",
                "three workstations beneath catwalk",
                dependencies=["create_catwalk"],
                expected_roles=["workstation"],
            ),
            IntentTask(
                "route_cables",
                "Route cable bundles through the bay.",
                "visible cable bundles",
                dependencies=["create_workstations"],
                expected_roles=["cable"],
            ),
            IntentTask(
                "mark_hazards",
                "Add hazard stripes at floor/catwalk edges.",
                "hazard stripes",
                dependencies=["create_floor", "create_catwalk"],
                expected_roles=["hazard_stripe"],
            ),
            IntentTask(
                "place_lights",
                "Add overhead area lighting.",
                "overhead lights",
                dependencies=["create_catwalk"],
                expected_roles=["overhead_light"],
            ),
            IntentTask(
                "frame_camera",
                "Frame a corridor camera on the completed bay.",
                "camera framing",
                dependencies=["create_floor", "create_catwalk", "create_workstations"],
                expected_roles=["camera"],
            ),
        ]
        dependencies = [
            Dependency("create_floor", "create_catwalk", "Catwalk needs a grounded frame of reference."),
            Dependency("create_catwalk", "create_workstations", "Workstations are positioned under the catwalk."),
            Dependency("create_workstations", "route_cables", "Cables attach meaningfully after equipment exists."),
            Dependency("create_catwalk", "place_lights", "Lights are placed over the built bay volume."),
            Dependency("create_workstations", "frame_camera", "Camera framing needs target content."),
        ]
        return IntentGraph(goal=IntentGoal("goal_maintenance_bay", prompt, BENCHMARK_NAME, tasks), dependencies=dependencies)

    def decompose(self, intent_graph: IntentGraph) -> list[PlanStep]:
        del intent_graph
        return [
            PlanStep("create_floor", "skill.build_floor", {}, [], ["role:floor"]),
            PlanStep("create_catwalk", "skill.build_raised_catwalk", {}, ["create_floor"], ["role:catwalk"]),
            PlanStep("create_workstations", "skill.build_workstation_row", {}, ["create_catwalk"], ["role:workstation count>=3"]),
            PlanStep("route_cables", "skill.build_cable_bundle", {}, ["create_workstations"], ["role:cable"]),
            PlanStep("mark_hazards", "skill.build_hazard_stripes", {}, ["create_floor", "create_catwalk"], ["role:hazard_stripe"]),
            PlanStep("place_lights", "skill.build_overhead_lights", {}, ["create_catwalk"], ["role:overhead_light"]),
            PlanStep("frame_camera", "skill.frame_corridor_camera", {}, ["create_workstations"], ["role:camera"]),
        ]

    def relevant_subgraph(self, world_graph: WorldGraph, step: PlanStep) -> WorldGraph:
        role_markers = {
            "skill.build_floor": {"floor"},
            "skill.build_raised_catwalk": {"floor", "catwalk"},
            "skill.build_workstation_row": {"floor", "catwalk", "workstation"},
            "skill.build_cable_bundle": {"workstation", "cable"},
            "skill.build_hazard_stripes": {"floor", "catwalk", "hazard_stripe"},
            "skill.build_overhead_lights": {"catwalk", "overhead_light"},
            "skill.frame_corridor_camera": {"floor", "catwalk", "workstation", "camera"},
        }.get(step.skill_name, set())
        nodes = {
            node_id: node
            for node_id, node in world_graph.nodes.items()
            if node.role in role_markers
        }
        relations = [
            relation
            for relation in world_graph.relations
            if relation.source in nodes and relation.target in nodes
        ]
        return WorldGraph(nodes=nodes, relations=relations)

    def completed_step_ids(self, world_graph: WorldGraph) -> set[str]:
        completed: set[str] = set()
        if self._has_horizontal_floor(world_graph):
            completed.add("create_floor")
        if world_graph.nodes_by_role("catwalk"):
            completed.add("create_catwalk")
        if len(world_graph.nodes_by_role("workstation")) >= 3:
            completed.add("create_workstations")
        if world_graph.nodes_by_role("cable"):
            completed.add("route_cables")
        if world_graph.nodes_by_role("hazard_stripe"):
            completed.add("mark_hazards")
        if world_graph.nodes_by_role("overhead_light"):
            completed.add("place_lights")
        if world_graph.nodes_by_role("camera"):
            completed.add("frame_camera")
        return completed

    def next_ready_steps(self, plan: list[PlanStep], completed: set[str], *, max_steps: int = 3) -> list[PlanStep]:
        ready: list[PlanStep] = []
        for step in plan:
            if step.id in completed:
                continue
            if all(dependency in completed or any(candidate.id == dependency for candidate in ready) for dependency in step.dependencies):
                ready.append(step)
            if len(ready) >= max_steps:
                break
        return ready

    def build_reasoning_step(self, context: AgentContext) -> ReasoningStep:
        world_graph = self.build_world_graph(context.scene_state)
        semantic_graph = self.build_semantic_graph(world_graph)
        intent_graph = self.build_intent_graph(context.user_prompt)
        plan = self.decompose(intent_graph)
        completed = self.completed_step_ids(world_graph)
        validation = validate_maintenance_bay_graph(world_graph)

        if validation.ok:
            report = PlanExecutionReport(
                executed_steps=sorted(completed),
                validation_results=validation,
                remaining_obligations=[],
            )
            return ReasoningStep(
                status="complete",
                narration="Maintenance-bay benchmark obligations are satisfied.",
                understanding="The graph validator found the floor, catwalk, workstations, cables, hazard stripes, lights, and camera.",
                plan=["stop after graph validation passes"],
                intended_actions=[],
                expected_outcome="The scene is ready for inspection.",
                preferred_execution_mode=context.execution_mode,
                continue_loop=False,
                metadata=self._metadata(world_graph, semantic_graph, intent_graph, report),
            )

        ready_steps = self.next_ready_steps(plan, completed)
        if not ready_steps:
            report = PlanExecutionReport(
                executed_steps=sorted(completed),
                validation_results=validation,
                remaining_obligations=intent_graph.missing_obligations(completed),
            )
            return ReasoningStep(
                status="error",
                narration="Planner blocked before emitting unsafe freeform edits.",
                understanding="The graph planner could not find a dependency-ready deterministic skill.",
                plan=["return a structured blocker instead of salvaging a single primitive"],
                intended_actions=[],
                expected_outcome="No scene damage should occur; the blocker identifies missing obligations.",
                preferred_execution_mode=context.execution_mode,
                continue_loop=False,
                error="maintenance_bay_planner_blocked",
                metadata=self._metadata(world_graph, semantic_graph, intent_graph, report),
            )

        actions = [
            {
                "action_id": step.id,
                "tool": step.skill_name,
                "params": dict(step.inputs),
            }
            for step in ready_steps
        ]
        executed = sorted(completed)
        report = PlanExecutionReport(
            executed_steps=executed,
            validation_results=validation,
            remaining_obligations=intent_graph.missing_obligations(completed),
        )
        return ReasoningStep(
            status="ok",
            narration=f"Executing deterministic maintenance-bay skills: {', '.join(step.id for step in ready_steps)}.",
            understanding="The prompt matches the maintenance-bay benchmark, so the hierarchical graph planner bypassed model-invented tool calls.",
            plan=[
                f"{step.id}: {step.description if hasattr(step, 'description') else step.skill_name}"
                for step in intent_graph.goal.tasks
            ],
            intended_actions=[f"{action['tool']}({action['params']})" for action in actions],
            expected_outcome="The next graph obligations are created with Z-up deterministic Blender skills.",
            preferred_execution_mode=context.execution_mode,
            continue_loop=True,
            assumptions=[
                {
                    "key": "coordinate_system",
                    "value": "Z-up",
                    "reason": "Blender uses Z as vertical; floor thickness and catwalk height are validated on Z.",
                },
                {
                    "key": "benchmark",
                    "value": BENCHMARK_NAME,
                    "reason": "The prompt asks for the maintenance-bay scene elements this vertical slice owns.",
                },
            ],
            metadata={
                **self._metadata(world_graph, semantic_graph, intent_graph, report),
                "actions": actions,
            },
        )

    @staticmethod
    def _has_horizontal_floor(world_graph: WorldGraph) -> bool:
        for floor in world_graph.nodes_by_role("floor"):
            dims = floor.dimensions
            if dims is not None and dims[2] <= min(dims[0], dims[1]) * 0.2:
                return True
        return False

    @staticmethod
    def _metadata(
        world_graph: WorldGraph,
        semantic_graph: SemanticGraph,
        intent_graph: IntentGraph,
        report: PlanExecutionReport,
    ) -> dict[str, Any]:
        validation = report.validation_results
        return {
            "planner": "maintenance_bay_htn_v1",
            "benchmark": BENCHMARK_NAME,
            "world_graph": {
                "nodes": [
                    {
                        "id": node.id,
                        "name": node.name,
                        "kind": node.kind,
                        "role": node.role,
                        "location": node.location,
                        "dimensions": node.dimensions,
                    }
                    for node in world_graph.nodes.values()
                ],
                "relation_count": len(world_graph.relations),
            },
            "semantic_graph": {
                "nodes": [
                    {"id": node.id, "label": node.label, "role": node.role}
                    for node in semantic_graph.nodes.values()
                ]
            },
            "intent_graph": {
                "goal": intent_graph.goal.id,
                "tasks": [
                    {
                        "id": task.id,
                        "obligation": task.obligation,
                        "dependencies": task.dependencies,
                    }
                    for task in intent_graph.goal.tasks
                ],
                "dependencies": [
                    {"before": dependency.before, "after": dependency.after, "reason": dependency.reason}
                    for dependency in intent_graph.dependencies
                ],
            },
            "plan_execution_report": {
                "executed_steps": report.executed_steps,
                "failed_steps": report.failed_steps,
                "remaining_obligations": report.remaining_obligations,
                "validation": None
                if validation is None
                else {
                    "ok": validation.ok,
                    "passed": validation.passed,
                    "failures": validation.failures,
                    "repairs": validation.repairs,
                },
            },
        }


def plan_maintenance_bay_step(context: AgentContext) -> ReasoningStep | None:
    if not is_maintenance_bay_prompt(context.user_prompt):
        return None
    return MaintenanceBayPlanner().build_reasoning_step(context)
