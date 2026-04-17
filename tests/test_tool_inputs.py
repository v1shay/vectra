from __future__ import annotations

from types import SimpleNamespace

import pytest

import vectra.tools.mesh_tools as mesh_tools_module
import vectra.tools.object_tools as object_tools_module
import vectra.tools.transform_tools as transform_tools_module
from vectra.tools.base import ToolValidationError
from vectra.tools.camera_tools import EnsureCameraTool
from vectra.tools.light_tools import CreateLightTool
from vectra.tools.mesh_tools import CreatePrimitiveTool
from vectra.tools.object_tools import DuplicateObjectTool
from vectra.tools.transform_tools import TransformObjectTool


def test_create_primitive_accepts_type_alias_and_defaults_location() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params({"primitive_type": "plane"})

    assert validated == {
        "type": "plane",
        "name": None,
        "location": (0.0, 0.0, 0.0),
        "scale": None,
        "rotation": None,
    }


def test_create_primitive_rejects_invalid_scale() -> None:
    tool = CreatePrimitiveTool()

    with pytest.raises(ToolValidationError, match="'scale' must be a 3-item list or tuple"):
        tool.validate_params({"type": "cube", "scale": {"x": 1}})


def test_create_primitive_rejects_non_finite_location() -> None:
    tool = CreatePrimitiveTool()

    with pytest.raises(ToolValidationError, match="'location' values must be finite"):
        tool.validate_params({"type": "cube", "location": [0.0, float("nan"), 0.0]})


def test_create_primitive_accepts_rotation_euler_alias() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params({"type": "cube", "rotation_euler": [0, 0, 1]})

    assert validated["rotation"] == (0.0, 0.0, 1.0)


def test_camera_ensure_accepts_rotation_euler_alias() -> None:
    tool = EnsureCameraTool()

    validated = tool.validate_params({"location": [1, 2, 3], "rotation_euler": [0.1, 0.2, 0.3]})

    assert validated == {
        "location": (1.0, 2.0, 3.0),
        "rotation": (0.1, 0.2, 0.3),
    }


def test_light_create_accepts_rotation_euler_alias() -> None:
    tool = CreateLightTool()

    validated = tool.validate_params({"rotation_euler": [0.1, 0.2, 0.3]})

    assert validated["rotation"] == (0.1, 0.2, 0.3)


def test_create_primitive_execute_applies_scale_and_rotation(monkeypatch) -> None:
    tool = CreatePrimitiveTool()
    created_object = SimpleNamespace(name="Cube", scale=None, rotation_euler=None)
    context = SimpleNamespace(active_object=None, mode="OBJECT")

    def primitive_cube_add(*, location):
        assert location == (0.0, 0.0, 0.0)
        context.active_object = created_object
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(mesh=SimpleNamespace(primitive_cube_add=primitive_cube_add)),
        context=SimpleNamespace(active_object=None, object=None),
    )
    monkeypatch.setattr(mesh_tools_module, "bpy", fake_bpy)
    monkeypatch.setattr(mesh_tools_module, "ensure_object_mode", lambda context: None)

    result = tool.execute(
        context,
        {"type": "cube", "scale": [2, 2, 2], "rotation": [0, 0, 1]},
    )

    assert created_object.scale == (2.0, 2.0, 2.0)
    assert created_object.rotation_euler == (0.0, 0.0, 1.0)
    assert result.outputs["object_name"] == "Cube"


def test_transform_validate_accepts_target_and_delta() -> None:
    tool = TransformObjectTool()

    validated = tool.validate_params({"target": "Cube", "delta": [1, 0, 0]})

    assert validated == {
        "target": "Cube",
        "delta": (1.0, 0.0, 0.0),
    }


def test_transform_execute_uses_resolved_target(monkeypatch) -> None:
    tool = TransformObjectTool()
    cube = SimpleNamespace(
        name="Cube",
        location=(1.0, 2.0, 3.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    fake_bpy = SimpleNamespace(
        context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)),
    )
    monkeypatch.setattr(transform_tools_module, "bpy", fake_bpy)
    monkeypatch.setattr(transform_tools_module, "resolve_object", lambda context, target: cube)

    result = tool.execute(context=None, params={"target": "Cube", "delta": [1, 0, 0]})

    assert cube.location == (2.0, 2.0, 3.0)
    assert result.outputs["object_name"] == "Cube"


def test_duplicate_validate_defaults_count() -> None:
    tool = DuplicateObjectTool()

    validated = tool.validate_params({"target": "Cube"})

    assert validated["count"] == 1
