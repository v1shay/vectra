from __future__ import annotations

import pytest

from agent_runtime.action_planner import ActionPlanningError, plan_actions
from agent_runtime.intent import IntentEnvelope, IntentStep, normalize_intent


def _single_cube_scene() -> dict[str, object]:
    return {
        "active_object": "Cube",
        "selected_objects": ["Cube"],
        "current_frame": 1,
        "objects": [
            {
                "name": "Cube",
                "type": "MESH",
                "selected": True,
                "active": True,
                "location": [0.0, 0.0, 0.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        ],
    }


def test_normalize_intent_maps_a_couple_into_downward_transform() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(
                action="transform",
                target="that plane",
                target_name="Plane",
                direction="into the ground",
                magnitude_qualifier="a couple",
                transform_kind="location",
                confidence=0.8,
            )
        ],
    )
    scene_state = {
        "active_object": "Plane",
        "selected_objects": ["Plane"],
        "current_frame": 1,
        "objects": [
            {
                "name": "Plane",
                "type": "MESH",
                "selected": True,
                "active": True,
                "location": [5.0, -11.0, 0.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        ],
    }

    normalized = normalize_intent(
        intent,
        prompt="move that plane a couple units into the ground",
        scene_state=scene_state,
        minimum_confidence=0.35,
    )

    assert normalized.status == "ok"
    assert normalized.steps[0].direction == "down"
    assert normalized.steps[0].magnitude == 2.0
    assert normalized.steps[0].target_name == "Plane"


def test_normalize_intent_resolves_pronoun_to_active_object() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(
                action="transform",
                target="it",
                direction="back",
                magnitude=2,
                transform_kind="location",
                confidence=0.8,
            )
        ],
    )

    normalized = normalize_intent(
        intent,
        prompt="move it back",
        scene_state=_single_cube_scene(),
        minimum_confidence=0.35,
    )

    assert normalized.status == "ok"
    assert normalized.steps[0].target_name == "Cube"
    assert normalized.steps[0].direction == "backward"


def test_normalize_intent_fails_closed_when_direction_is_missing() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(
                action="transform",
                target_name="Cube",
                magnitude_qualifier="a bit",
                transform_kind="location",
                confidence=0.8,
            )
        ],
    )

    normalized = normalize_intent(
        intent,
        prompt="move cube a bit",
        scene_state=_single_cube_scene(),
        minimum_confidence=0.35,
    )

    assert normalized.status == "no_action"
    assert "direction" in normalized.reason.lower()


def test_plan_actions_builds_absolute_transform_vector() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(
                action="transform",
                target_name="Cube",
                direction="forward",
                magnitude=2.0,
                transform_kind="location",
                confidence=0.8,
            )
        ],
    )

    actions = plan_actions(intent, scene_state=_single_cube_scene())

    assert actions == [
        {
            "action_id": "step_1_transform_cube",
            "tool": "object.transform",
            "params": {"object_name": "Cube", "location": [0.0, 2.0, 0.0]},
        }
    ]


def test_plan_actions_builds_rotation_with_default_z_axis() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(
                action="transform",
                target_name="Cube",
                magnitude=45.0,
                transform_kind="rotation",
                axis="z",
                confidence=0.8,
            )
        ],
    )

    actions = plan_actions(intent, scene_state=_single_cube_scene())

    assert actions[0]["tool"] == "object.transform"
    assert actions[0]["params"]["rotation_euler"][2] == pytest.approx(0.78539816339)


def test_plan_actions_supports_previous_step_refs_for_multi_step_requests() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(action="create", primitive_type="cube", confidence=0.8),
            IntentStep(
                action="transform",
                target_ref="previous_step",
                target="it",
                direction="forward",
                magnitude=2.0,
                transform_kind="location",
                confidence=0.8,
            ),
        ],
    )

    actions = plan_actions(intent, scene_state={"active_object": None, "selected_objects": [], "objects": []})

    assert actions == [
        {
            "action_id": "step_1_create_cube",
            "tool": "mesh.create_primitive",
            "params": {"primitive_type": "cube", "location": [0.0, 0.0, 0.0]},
        },
        {
            "action_id": "step_2_transform_previous",
            "tool": "object.transform",
            "params": {
                "object_name": {"$ref": "step_1_create_cube.object_name"},
                "location": [0.0, 2.0, 0.0],
            },
        },
    ]


def test_plan_actions_rejects_unresolved_previous_step_reference() -> None:
    intent = IntentEnvelope(
        status="ok",
        confidence=0.8,
        reason="",
        steps=[
            IntentStep(
                action="transform",
                target_ref="previous_step",
                direction="forward",
                magnitude=2.0,
                transform_kind="location",
                confidence=0.8,
            )
        ],
    )

    with pytest.raises(ActionPlanningError, match="previous object"):
        plan_actions(intent, scene_state={"active_object": None, "selected_objects": [], "objects": []})
