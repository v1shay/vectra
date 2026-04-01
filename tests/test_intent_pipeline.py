from __future__ import annotations

from agent_runtime.construction import compile_construction_plan, decompose_scene_intent
from agent_runtime.scene_intent import (
    SceneEntity,
    SceneIntent,
    SceneRelationship,
    SceneTransformIntent,
    normalize_scene_intent,
    parse_scene_intent_content,
)


def test_parse_scene_intent_falls_back_to_labeled_text() -> None:
    content = """
    STATUS: ok
    CONFIDENCE: 0.9
    REASONING: Build two cubes and separate them.
    ENTITIES:
    - logical_id: cube_pair; kind: cube; quantity: 2; group_id: cube_pair_group
    RELATIONSHIPS:
    - logical_id: spread_pair; relation_type: relative_offset; source_id: cube_pair; offset: [3.0, 0.0, 0.0]
    GROUPS:
    - logical_id: cube_pair_group; entity_ids: ['cube_pair_1', 'cube_pair_2']; pronouns: ['them', 'both']
    """

    intent = parse_scene_intent_content(content)

    assert intent.status == "ok"
    assert intent.entities[0].logical_id == "cube_pair"
    assert intent.relationships[0].relation_type == "relative_offset"


def test_compile_construction_plan_handles_multiple_entities_with_absolute_offsets() -> None:
    intent = normalize_scene_intent(
        SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create two entities with fixed offsets.",
            entities=[
                SceneEntity(
                    logical_id="plane_left",
                    kind="plane",
                    initial_transform=SceneTransformIntent(offset=[3.0, 0.0, -10.0]),
                ),
                SceneEntity(
                    logical_id="cube_right",
                    kind="cube",
                    initial_transform=SceneTransformIntent(offset=[-17.0, 0.0, -4.0]),
                ),
            ],
        ),
        minimum_confidence=0.35,
    )

    compiled = compile_construction_plan(intent, scene_state={"objects": []}, max_construction_steps=None)

    assert compiled.actions == [
        {
            "action_id": "create_plane_left",
            "tool": "mesh.create_primitive",
            "params": {
                "primitive_type": "plane",
                "name": "Plane_1",
                "location": [3.0, 0.0, -10.0],
            },
        },
        {
            "action_id": "create_cube_right",
            "tool": "mesh.create_primitive",
            "params": {
                "primitive_type": "cube",
                "name": "Cube_1",
                "location": [-17.0, 0.0, -4.0],
            },
        },
    ]


def test_compile_construction_plan_creates_group_and_resolves_relative_offset() -> None:
    intent = normalize_scene_intent(
        SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create two cubes and move them apart.",
            entities=[
                SceneEntity(
                    logical_id="cube_pair",
                    kind="cube",
                    quantity=2,
                    group_id="cube_pair_group",
                )
            ],
            relationships=[
                SceneRelationship(
                    logical_id="spread_pair",
                    relation_type="relative_offset",
                    source_id="cube_pair",
                    offset=[3.0, 0.0, 0.0],
                )
            ],
        ),
        minimum_confidence=0.35,
    )

    compiled = compile_construction_plan(intent, scene_state={"objects": []}, max_construction_steps=None)

    assert [action["tool"] for action in compiled.actions] == [
        "mesh.create_primitive",
        "mesh.create_primitive",
        "object.transform",
    ]
    assert compiled.state.groups["cube_pair_group"] == ["cube_pair_1", "cube_pair_2"]
    assert compiled.actions[-1]["params"]["object_name"] == "Cube_2"
    assert compiled.actions[-1]["params"]["location"] == [3.0, 0.0, 0.0]


def test_compile_construction_plan_uses_deterministic_naming_from_scene_state() -> None:
    intent = normalize_scene_intent(
        SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create one more cube.",
            entities=[SceneEntity(logical_id="cube_pair", kind="cube", quantity=2, group_id="cube_pair_group")],
        ),
        minimum_confidence=0.35,
    )

    compiled = compile_construction_plan(
        intent,
        scene_state={
            "objects": [
                {
                    "name": "Cube_1",
                    "type": "MESH",
                    "location": [0.0, 0.0, 0.0],
                    "dimensions": [2.0, 2.0, 2.0],
                }
            ]
        },
        max_construction_steps=1,
    )

    assert compiled.actions[0]["params"]["name"] == "Cube_2"


def test_compile_construction_plan_satisfies_above_relationship_from_geometry() -> None:
    intent = normalize_scene_intent(
        SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Place one cube on top of another.",
            entities=[
                SceneEntity(logical_id="base_cube", kind="cube"),
                SceneEntity(logical_id="top_cube", kind="cube"),
            ],
            relationships=[
                SceneRelationship(
                    logical_id="stack_top",
                    relation_type="above",
                    source_id="top_cube",
                    target_id="base_cube",
                    metadata={"touching": True},
                )
            ],
        ),
        minimum_confidence=0.35,
    )

    compiled = compile_construction_plan(intent, scene_state={"objects": []}, max_construction_steps=None)

    assert compiled.actions[0]["params"]["name"] == "Cube_1"
    assert compiled.actions[1]["params"]["name"] == "Cube_2"
    assert compiled.actions[-1]["tool"] == "object.transform"
    assert compiled.actions[-1]["params"]["location"] == [0.0, 0.0, 2.0]


def test_compile_construction_plan_expands_staircase_pattern_into_stepwise_relations() -> None:
    intent = normalize_scene_intent(
        SceneIntent(
            status="ok",
            confidence=0.9,
            reasoning="Create a staircase of cubes.",
            entities=[
                SceneEntity(
                    logical_id="stairs",
                    kind="cube",
                    quantity=5,
                    group_id="stairs_group",
                )
            ],
            metadata={"pattern": "staircase", "stair_offset": [1.25, 0.0, 1.0]},
        ),
        minimum_confidence=0.35,
    )

    steps = decompose_scene_intent(intent)
    compiled = compile_construction_plan(intent, scene_state={"objects": []}, max_construction_steps=None)

    assert any(step.kind == "satisfy_relation" for step in steps)
    assert len([action for action in compiled.actions if action["tool"] == "mesh.create_primitive"]) == 5
    assert compiled.actions[-1]["params"]["location"] == [5.0, 0.0, 4.0]
