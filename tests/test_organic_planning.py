from __future__ import annotations

from agent_runtime.director.models import DirectorContext, ToolCall
from agent_runtime.director.organic import build_organic_scene_metadata


def _context(prompt: str) -> DirectorContext:
    return DirectorContext(
        user_prompt=prompt,
        scene_state={"objects": [{"name": "ExistingBase", "type": "MESH", "location": [0, 0, 0]}]},
        screenshot=None,
        history=[],
        iteration=1,
        execution_mode="vectra-dev",
    )


def _metadata_for(prompt: str, tool_calls: list[ToolCall]) -> dict[str, object]:
    actions = [
        {
            "action_id": f"step_{index}",
            "tool": tool_call.name,
            "params": {"name": tool_call.arguments.get("name", f"Object_{index}")},
        }
        for index, tool_call in enumerate(tool_calls, start=1)
    ]
    return build_organic_scene_metadata(
        context=_context(prompt),
        tool_calls=tool_calls,
        actions=actions,
        action_families=["create"],
    )


def test_organic_metadata_reflects_shape_prompt_without_template_names() -> None:
    metadata = _metadata_for(
        "Create a red cube on the left and a blue cylinder on the right.",
        [
            ToolCall("mesh.create_primitive", {"type": "cube", "name": "LeftCube"}),
            ToolCall("mesh.create_primitive", {"type": "cylinder", "name": "RightCylinder"}),
        ],
    )

    assert metadata["planning_mode"] == "organic_scene_graph_v1"
    assert metadata["hardcoding_policy"] == "clean"
    assert {"cube", "left", "blue", "cylinder", "right"}.issubset(metadata["semantic_terms"])
    assert "GeneratedInterior" not in str(metadata)
    assert "maintenance_bay_htn_v1" not in str(metadata)


def test_organic_task_graph_changes_with_prompt_specific_tools() -> None:
    kitchen = _metadata_for(
        "Create a compact kitchen counter corner with a sink basin, faucet, cutting board, and two mugs.",
        [
            ToolCall("mesh.create_primitive", {"type": "cube", "name": "Counter"}),
            ToolCall("mesh.create_primitive", {"type": "cylinder", "name": "Faucet"}),
            ToolCall("material.apply_basic", {"target": "Counter"}),
        ],
    )
    airport = _metadata_for(
        "Create a tiny airport security tray scene with one tray, two shoes, a laptop rectangle, and three bins.",
        [
            ToolCall("mesh.create_primitive", {"type": "cube", "name": "Tray"}),
            ToolCall("object.duplicate", {"target": "Tray"}),
            ToolCall("object.distribute", {"targets": ["Tray", "TrayCopy"]}),
        ],
    )

    assert kitchen["semantic_terms"] != airport["semantic_terms"]
    assert kitchen["organic_task_graph"] != airport["organic_task_graph"]
    assert "kitchen" in kitchen["semantic_terms"]
    assert "airport" in airport["semantic_terms"]
