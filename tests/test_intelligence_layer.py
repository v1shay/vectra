from __future__ import annotations

from pathlib import Path

from agent_runtime.agent.models import AgentContext
from agent_runtime.agent.reasoner import reason_step
from agent_runtime.director.loop import _salvageable_progress_batch
from agent_runtime.director.models import ToolCall, ToolCallValidationIssue
from agent_runtime.intelligence.graphs import world_graph_from_scene_state
from agent_runtime.intelligence.planner import MaintenanceBayPlanner
from agent_runtime.intelligence.validation import validate_maintenance_bay_graph
from agent_runtime.memory.providers.jsonl import JsonlMemoryProvider
from vectra.tools.registry import ToolRegistry


MAINTENANCE_PROMPT = "Create a maintenance bay with a raised catwalk, three workstations, cables, hazard stripes, lights, and camera"


def _context(scene_state: dict[str, object] | None = None, *, prompt: str = MAINTENANCE_PROMPT) -> AgentContext:
    return AgentContext(
        user_prompt=prompt,
        scene_state=scene_state or {"objects": [], "selected_objects": [], "active_object": None},
        screenshot=None,
        history=[],
        iteration=1,
        execution_mode="vectra-dev",
        memory_results=[],
    )


def test_maintenance_prompt_routes_to_htn_skills_without_director(monkeypatch) -> None:
    def fail_director(_context):
        raise AssertionError("maintenance-bay prompt should not call DirectorLoop")

    monkeypatch.setattr("agent_runtime.agent.reasoner._DIRECTOR_LOOP.step", fail_director)

    reasoning = reason_step(_context())

    assert reasoning.status == "ok"
    assert reasoning.metadata["planner"] == "maintenance_bay_htn_v1"
    assert [action["tool"] for action in reasoning.metadata["actions"]] == [
        "skill.build_floor",
        "skill.build_raised_catwalk",
        "skill.build_workstation_row",
    ]


def test_htn_decomposition_orders_floor_before_catwalk_and_workstations() -> None:
    planner = MaintenanceBayPlanner()
    intent_graph = planner.build_intent_graph(MAINTENANCE_PROMPT)
    plan = planner.decompose(intent_graph)
    index_by_step = {step.id: index for index, step in enumerate(plan)}

    assert index_by_step["create_floor"] < index_by_step["create_catwalk"]
    assert index_by_step["create_catwalk"] < index_by_step["create_workstations"]
    assert index_by_step["place_lights"] < index_by_step["frame_camera"]


def test_z_up_floor_validation_rejects_vertical_slab() -> None:
    world_graph = world_graph_from_scene_state(
        {
            "objects": [
                {
                    "name": "FloorBase",
                    "type": "MESH",
                    "location": [0, 0, 0],
                    "dimensions": [10, 0.1, 10],
                    "custom_properties": {"vectra_role": "floor"},
                }
            ]
        }
    )

    report = validate_maintenance_bay_graph(world_graph)

    assert not report.ok
    assert "missing horizontal Z-up floor" in report.failures


def test_single_action_salvage_is_rejected_after_batch_validation_failure() -> None:
    salvage = _salvageable_progress_batch(
        [ToolCall("mesh.create_primitive", {"type": "cube", "name": "FloorBase"})],
        [
            ToolCallValidationIssue(
                "mesh.create_primitive",
                "This prompt needs a coordinated batch of 2 to 4 tool calls rather than a single local action.",
            )
        ],
    )

    assert salvage == []


def test_completed_scene_validates_required_obligations() -> None:
    scene_state = {
        "objects": [
            {"name": "MaintenanceBay_Floor", "type": "MESH", "location": [0, 0, -0.05], "dimensions": [10, 6, 0.1]},
            {"name": "MaintenanceBay_Catwalk", "type": "MESH", "location": [0, 0, 2.2], "dimensions": [7, 1, 0.2]},
            {"name": "MaintenanceBay_Workstation_1", "type": "MESH", "location": [-2, 0, 0.5], "dimensions": [1, 1, 1]},
            {"name": "MaintenanceBay_Workstation_2", "type": "MESH", "location": [0, 0, 0.5], "dimensions": [1, 1, 1]},
            {"name": "MaintenanceBay_Workstation_3", "type": "MESH", "location": [2, 0, 0.5], "dimensions": [1, 1, 1]},
            {"name": "MaintenanceBay_CableBundle_Main", "type": "MESH", "location": [0, -1, 0.1], "dimensions": [5, 0.1, 0.1]},
            {"name": "MaintenanceBay_HazardStripe_Floor_1", "type": "MESH", "location": [0, -2, 0.1], "dimensions": [1, 0.1, 0.1]},
            {"name": "MaintenanceBay_OverheadLight_1", "type": "LIGHT", "location": [0, -1, 4], "dimensions": [1, 1, 1]},
            {"name": "MaintenanceBay_Camera", "type": "CAMERA", "location": [6, -6, 3], "dimensions": [1, 1, 1]},
        ]
    }

    reasoning = reason_step(_context(scene_state))

    assert reasoning.status == "complete"
    assert reasoning.continue_loop is False
    assert reasoning.metadata["plan_execution_report"]["validation"]["ok"] is True


def test_relevant_subgraph_filters_by_step_role() -> None:
    planner = MaintenanceBayPlanner()
    graph = world_graph_from_scene_state(
        {
            "objects": [
                {"name": "MaintenanceBay_Floor", "type": "MESH", "location": [0, 0, 0], "dimensions": [10, 6, 0.1]},
                {"name": "DecorativeCube", "type": "MESH", "location": [20, 20, 20], "dimensions": [1, 1, 1]},
            ]
        }
    )
    step = planner.decompose(planner.build_intent_graph(MAINTENANCE_PROMPT))[0]

    subgraph = planner.relevant_subgraph(graph, step)

    assert {node.name for node in subgraph.nodes.values()} == {"MaintenanceBay_Floor"}


def test_jsonl_memory_persists_and_retrieves_failure_signature(tmp_path: Path) -> None:
    provider = JsonlMemoryProvider(tmp_path / "memory.jsonl")
    provider.add_memory(
        {
            "prompt": MAINTENANCE_PROMPT,
            "benchmark": "maintenance_bay",
            "failure_signature": "missing horizontal Z-up floor",
            "summary": "failed validation",
        }
    )

    matches = provider.query_memory("maintenance_bay horizontal floor", top_k=1)

    assert matches[0]["failure_signature"] == "missing horizontal Z-up floor"


def test_deterministic_maintenance_tools_are_discoverable() -> None:
    registry = ToolRegistry()
    registry.discover()

    assert "skill.build_floor" in registry.list_tools()
    assert "skill.frame_corridor_camera" in registry.list_tools()
