from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import vectra.agent.observation as observation_module


class _CornerVector:
    def __init__(self, coords: tuple[float, float, float]) -> None:
        self._coords = coords

    def __iter__(self):
        return iter(self._coords)


class _TranslatedMatrix:
    def __matmul__(self, other):
        return (float(other[0]) + 10.0, float(other[1]) - 2.0, float(other[2]) + 1.0)


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
    assert scene_state["objects"][0]["spatial"]["center"] == [1.0, 2.0, 3.0]
    assert "top" in scene_state["objects"][0]["spatial"]["face_centers"]
    assert scene_state["spatial_anchors"]
    assert scene_state["spatial_relations"] == []


def test_build_scene_state_uses_world_bounds_for_spatial_observation(monkeypatch) -> None:
    obj = SimpleNamespace(
        name="WorldBoundedCube",
        type="MESH",
        select_get=lambda: False,
        location=(10.0, -2.0, 1.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        dimensions=(100.0, 100.0, 100.0),
        bound_box=[
            _CornerVector((-1.0, -1.0, -1.0)),
            _CornerVector((-1.0, -1.0, 1.0)),
            _CornerVector((-1.0, 1.0, -1.0)),
            _CornerVector((-1.0, 1.0, 1.0)),
            _CornerVector((1.0, -1.0, -1.0)),
            _CornerVector((1.0, -1.0, 1.0)),
            _CornerVector((1.0, 1.0, -1.0)),
            _CornerVector((1.0, 1.0, 1.0)),
        ],
        matrix_world=_TranslatedMatrix(),
        parent=None,
        children=[],
        users_collection=[],
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

    assert scene_state["objects"][0]["bounds"] == {
        "min": [9.0, -3.0, 0.0],
        "max": [11.0, -1.0, 2.0],
    }
    assert scene_state["objects"][0]["spatial"]["center"] == [10.0, -2.0, 1.0]


def test_build_scene_state_includes_stable_spatial_relations_and_anchors(monkeypatch) -> None:
    floor = SimpleNamespace(
        name="Floor",
        type="MESH",
        select_get=lambda: False,
        location=(0.0, 0.0, 0.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        dimensions=(6.0, 6.0, 0.0),
        parent=None,
        children=[],
        users_collection=[],
    )
    cube = SimpleNamespace(
        name="Cube",
        type="MESH",
        select_get=lambda: False,
        location=(0.0, 0.0, 1.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        dimensions=(2.0, 2.0, 2.0),
        parent=None,
        children=[],
        users_collection=[],
    )
    fake_bpy = SimpleNamespace()
    monkeypatch.setattr(observation_module, "bpy", fake_bpy)

    context = SimpleNamespace(
        scene=SimpleNamespace(frame_current=1, objects=[cube, floor]),
        active_object=None,
        selected_objects=[],
    )
    first_state = observation_module.build_scene_state(context)
    second_state = observation_module.build_scene_state(context)

    assert {"source": "Cube", "target": "Floor", "relation": "on"} in first_state["spatial_relations"]
    assert first_state["objects"][0]["spatial"]["grounded"] is True
    assert first_state["objects"][0]["spatial"]["floor_contact"] == {"object": "Floor", "gap": 0.0}
    assert first_state["spatial_relations"] == second_state["spatial_relations"]
    assert first_state["spatial_anchors"] == second_state["spatial_anchors"]


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
