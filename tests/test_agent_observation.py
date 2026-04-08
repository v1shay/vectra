from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import vectra.agent.observation as observation_module


def test_build_scene_state_includes_relationship_and_dimension_fields(monkeypatch) -> None:
    parent = SimpleNamespace(name="Parent")
    child = SimpleNamespace(name="Child")
    obj = SimpleNamespace(
        name="Cube",
        type="MESH",
        select_get=lambda: True,
        location=(1.0, 2.0, 3.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        dimensions=(2.0, 2.0, 2.0),
        parent=parent,
        children=[child],
        users_collection=[SimpleNamespace(name="Collection")],
    )
    fake_bpy = SimpleNamespace()
    monkeypatch.setattr(observation_module, "bpy", fake_bpy)

    scene_state = observation_module.build_scene_state(
        SimpleNamespace(
            scene=SimpleNamespace(frame_current=1, objects=[obj]),
            active_object=obj,
            selected_objects=[obj],
        )
    )

    assert scene_state["objects"][0]["dimensions"] == [2.0, 2.0, 2.0]
    assert scene_state["objects"][0]["parent"] == "Parent"
    assert scene_state["objects"][0]["children"] == ["Child"]
    assert scene_state["objects"][0]["collection_names"] == ["Collection"]


def test_capture_viewport_screenshot_returns_unavailable_when_no_view3d_area(monkeypatch) -> None:
    fake_bpy = SimpleNamespace(
        context=SimpleNamespace(
            scene=SimpleNamespace(render=SimpleNamespace(filepath="")),
            window_manager=SimpleNamespace(windows=[]),
            temp_override=lambda **kwargs: nullcontext(),
        ),
        ops=SimpleNamespace(render=SimpleNamespace(opengl=lambda **kwargs: {"FINISHED"})),
    )
    monkeypatch.setattr(observation_module, "bpy", fake_bpy)

    result = observation_module.capture_viewport_screenshot(1)

    assert result["available"] is False
    assert "VIEW_3D" in result["reason"]


def test_build_scene_state_includes_animation_summary(monkeypatch) -> None:
    keyframe_points = [
        SimpleNamespace(co=(1.0, 0.0)),
        SimpleNamespace(co=(24.0, 2.5)),
    ]
    animation_curve = SimpleNamespace(data_path="location", keyframe_points=keyframe_points)
    obj = SimpleNamespace(
        name="AnimatedCube",
        type="MESH",
        select_get=lambda: False,
        location=(0.0, 0.0, 0.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        dimensions=(2.0, 2.0, 2.0),
        parent=None,
        children=[],
        users_collection=[],
        animation_data=SimpleNamespace(action=SimpleNamespace(fcurves=[animation_curve])),
    )
    fake_bpy = SimpleNamespace()
    monkeypatch.setattr(observation_module, "bpy", fake_bpy)

    scene_state = observation_module.build_scene_state(
        SimpleNamespace(
            scene=SimpleNamespace(frame_current=1, objects=[obj]),
            active_object=None,
            selected_objects=[],
        )
    )

    animation_summary = scene_state["objects"][0]["animation_summary"]
    assert animation_summary["frame_start"] == 1
    assert animation_summary["frame_end"] == 24
    assert animation_summary["animated_properties"] == ["location"]
    assert animation_summary["visible_motion"] is True
