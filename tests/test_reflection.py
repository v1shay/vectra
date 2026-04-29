from __future__ import annotations

from vectra.agent.reflection import summarize_scene_diff


def test_partial_structural_room_progress_counts_as_meaningful_change() -> None:
    before = {
        "current_frame": 1,
        "scene_bounds": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
        "lights": [],
        "active_camera": None,
        "groups": [],
        "objects": [],
    }
    after = {
        "current_frame": 1,
        "scene_bounds": {"min": [-2.0, -2.0, -0.1], "max": [2.0, 2.0, 2.5]},
        "lights": [],
        "active_camera": None,
        "groups": [{"name": "RoomShell", "object_names": ["Floor", "BackWall"]}],
        "objects": [
            {
                "name": "Floor",
                "location": [0.0, 0.0, 0.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "material_names": [],
                "parent": None,
                "collection_names": ["RoomShell"],
                "keyframe_count": 0,
                "visible_animation": False,
            },
            {
                "name": "BackWall",
                "location": [0.0, 2.0, 1.2],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "material_names": [],
                "parent": None,
                "collection_names": ["RoomShell"],
                "keyframe_count": 0,
                "visible_animation": False,
            },
        ],
    }

    diff = summarize_scene_diff(before, after, {"metadata": {"action_families": ["create", "structure"]}})

    assert diff["structural_progress"] is True
    assert diff["meaningful_change"] is True
    assert diff["progress_score"] >= 0.75
    assert "added grouping" in diff["progress_reasons"]


def test_visible_animation_requires_frame_span_and_motion_signal() -> None:
    before = {
        "current_frame": 1,
        "scene_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        "lights": [{"name": "KeyLight"}],
        "active_camera": "Camera",
        "groups": [],
        "objects": [
            {
                "name": "KeyLight",
                "location": [4.0, -4.0, 6.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "material_names": [],
                "parent": None,
                "collection_names": [],
                "keyframe_count": 0,
                "visible_animation": False,
            }
        ],
    }
    after = {
        "current_frame": 1,
        "scene_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        "lights": [{"name": "KeyLight"}],
        "active_camera": "Camera",
        "groups": [],
        "objects": [
            {
                "name": "KeyLight",
                "location": [4.0, -4.0, 6.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "material_names": [],
                "parent": None,
                "collection_names": [],
                "keyframe_count": 4,
                "visible_animation": True,
            }
        ],
    }

    diff = summarize_scene_diff(before, after, {"metadata": {"action_families": ["animation", "light"]}})

    assert diff["visible_animation"] is True
    assert diff["added_visible_animation_objects"] == ["KeyLight"]
    assert diff["meaningful_change"] is True
    assert "added visible motion" in diff["progress_reasons"]


def test_spatial_diagnostics_penalize_structural_progress_when_scene_gets_worse() -> None:
    before = {
        "current_frame": 1,
        "scene_bounds": {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]},
        "lights": [],
        "active_camera": None,
        "groups": [],
        "objects": [
            {
                "name": "Floor",
                "location": [0.0, 0.0, 0.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "material_names": [],
                "parent": None,
                "collection_names": [],
                "keyframe_count": 0,
                "visible_animation": False,
                "spatial_diagnostics": {"issues": [], "severity": 0},
            }
        ],
    }
    after = {
        "current_frame": 1,
        "scene_bounds": {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 3.0]},
        "lights": [],
        "active_camera": None,
        "groups": [],
        "objects": [
            before["objects"][0],
            {
                "name": "FloatingBox",
                "location": [0.0, 0.0, 2.5],
                "rotation_euler": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "material_names": [],
                "parent": None,
                "collection_names": [],
                "keyframe_count": 0,
                "visible_animation": False,
                "spatial_diagnostics": {"issues": ["floating", "unsupported"], "severity": 4},
            },
        ],
    }

    diff = summarize_scene_diff(before, after, {"metadata": {"action_families": ["create", "structure"]}})

    assert diff["structural_progress"] is False
    assert diff["new_spatial_issue_count"] == 2
    assert "penalized new spatial issue(s)" in diff["progress_reasons"]
    assert "FloatingBox:floating" in diff["summary"]
