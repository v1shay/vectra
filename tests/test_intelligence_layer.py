from __future__ import annotations

import importlib.util

from agent_runtime.intelligence import WorldGraph, world_graph_from_scene_state
from vectra.tools.registry import ToolRegistry


def test_intelligence_package_exposes_generic_graph_primitives_only() -> None:
    graph = world_graph_from_scene_state(
        {
            "objects": [
                {"name": "Floor", "type": "MESH", "location": [0, 0, 0], "dimensions": [4, 4, 0.1]},
                {"name": "Camera", "type": "CAMERA", "location": [4, -4, 3]},
            ]
        }
    )

    assert isinstance(graph, WorldGraph)
    assert graph.names() == {"Floor", "Camera"}


def test_hardcoded_benchmark_intelligence_modules_are_removed() -> None:
    assert importlib.util.find_spec("agent_runtime.intelligence.planner") is None
    assert importlib.util.find_spec("agent_runtime.intelligence.validation") is None


def test_template_builder_tool_modules_are_removed_from_runtime() -> None:
    assert importlib.util.find_spec("vectra.tools.maintenance_bay_tools") is None
    assert importlib.util.find_spec("vectra.tools.composition_tools") is None

    registry = ToolRegistry()
    registry.discover()
    discovered_tools = registry.list_tools()

    assert all(not tool_name.startswith("skill.build_") for tool_name in discovered_tools)
    assert "scene.build_room_shell" not in discovered_tools
    assert "scene.build_focal_furniture" not in discovered_tools


def test_legacy_hardcoded_intent_pipelines_are_removed() -> None:
    assert importlib.util.find_spec("agent_runtime.intent") is None
    assert importlib.util.find_spec("agent_runtime.action_planner") is None
    assert importlib.util.find_spec("agent_runtime.construction") is None
    assert importlib.util.find_spec("agent_runtime.scene_intent") is None
    assert importlib.util.find_spec("agent_runtime.scene_pipeline") is None
