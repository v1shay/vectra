from __future__ import annotations

from types import SimpleNamespace

import pytest

import vectra.tools.mesh_tools as mesh_tools_module
import vectra.tools.transform_tools as transform_tools_module
from vectra.tools.base import ToolValidationError
from vectra.tools.mesh_tools import CreatePrimitiveTool
from vectra.tools.transform_tools import TransformObjectTool


def test_create_primitive_rejects_null_optional_fields() -> None:
    tool = CreatePrimitiveTool()

    with pytest.raises(ToolValidationError, match="'location' must be a 3-item list or tuple"):
        tool.validate_params(
            {
                "primitive_type": "uv_sphere",
                "location": None,
                "name": None,
            }
        )


def test_create_primitive_rejects_blank_name() -> None:
    tool = CreatePrimitiveTool()

    with pytest.raises(ToolValidationError, match="'name' must be a non-empty string when provided"):
        tool.validate_params(
            {
                "primitive_type": "plane",
                "location": [0, 0, 0],
                "name": "",
            }
        )


def test_create_primitive_rejects_mapping_location() -> None:
    tool = CreatePrimitiveTool()

    with pytest.raises(ToolValidationError, match="'location' must be a 3-item list or tuple"):
        tool.validate_params(
            {
                "primitive_type": "uv_sphere",
                "location": {"x": 10},
            }
        )


def test_create_primitive_accepts_exact_params() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params(
        {
            "primitive_type": "plane",
            "location": [1, 2, 3],
            "name": "Ground",
        }
    )

    assert validated == {
        "primitive_type": "plane",
        "name": "Ground",
        "location": (1.0, 2.0, 3.0),
    }


def test_create_primitive_execute_uses_exact_location(monkeypatch) -> None:
    tool = CreatePrimitiveTool()
    context = SimpleNamespace(active_object=None, mode="OBJECT")
    calls: dict[str, object] = {}

    def primitive_uv_sphere_add(*, location):
        calls["location"] = location
        context.active_object = SimpleNamespace(name="Sphere")
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(
            mesh=SimpleNamespace(primitive_uv_sphere_add=primitive_uv_sphere_add)
        ),
        context=SimpleNamespace(active_object=None, object=None),
    )
    monkeypatch.setattr(mesh_tools_module, "bpy", fake_bpy)

    result = tool.execute(
        context,
        {
            "primitive_type": "uv_sphere",
            "location": [0, 0, 0],
            "name": "Sphere",
        },
    )

    assert calls["location"] == (0.0, 0.0, 0.0)
    assert result.outputs == {"object_name": "Sphere"}


def test_create_primitive_execute_switches_to_object_mode(monkeypatch) -> None:
    tool = CreatePrimitiveTool()
    context = SimpleNamespace(active_object=None, mode="EDIT_MESH")
    calls: dict[str, object] = {}

    def mode_set(*, mode):
        calls["mode"] = mode
        return {"FINISHED"}

    def primitive_cube_add(*, location):
        calls["location"] = location
        context.active_object = SimpleNamespace(name="Cube")
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(
            object=SimpleNamespace(mode_set=mode_set),
            mesh=SimpleNamespace(primitive_cube_add=primitive_cube_add),
        ),
        context=SimpleNamespace(active_object=None, object=None),
    )
    monkeypatch.setattr(mesh_tools_module, "bpy", fake_bpy)

    result = tool.execute(context, {"primitive_type": "cube"})

    assert calls["mode"] == "OBJECT"
    assert calls["location"] == (0.0, 0.0, 0.0)
    assert result.outputs == {"object_name": "Cube"}


def test_create_primitive_execute_falls_back_to_bpy_context_active_object(monkeypatch) -> None:
    tool = CreatePrimitiveTool()
    created_object = SimpleNamespace(name="Plane")
    context = SimpleNamespace(active_object=None, mode="OBJECT")

    def primitive_plane_add(*, location):
        del location
        fake_bpy.context.active_object = created_object
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(
            mesh=SimpleNamespace(primitive_plane_add=primitive_plane_add)
        ),
        context=SimpleNamespace(active_object=None, object=None),
    )
    monkeypatch.setattr(mesh_tools_module, "bpy", fake_bpy)

    result = tool.execute(context, {"primitive_type": "plane"})

    assert result.outputs == {"object_name": "Plane"}


def test_transform_validate_rejects_partial_mapping() -> None:
    tool = TransformObjectTool()

    with pytest.raises(ToolValidationError, match="'location' must be a 3-item list or tuple"):
        tool.validate_params(
            {
                "object_name": "Cube",
                "location": {"x": 10},
            }
        )


def test_transform_validate_rejects_null_location() -> None:
    tool = TransformObjectTool()

    with pytest.raises(ToolValidationError, match="'location' must be a 3-item list or tuple"):
        tool.validate_params(
            {
                "object_name": "Cube",
                "location": None,
            }
        )


def test_transform_validate_accepts_exact_vector() -> None:
    tool = TransformObjectTool()

    validated = tool.validate_params(
        {
            "object_name": "Cube",
            "location": [10, 2, 3],
        }
    )

    assert validated == {
        "object_name": "Cube",
        "location": (10.0, 2.0, 3.0),
    }


def test_transform_execute_sets_exact_location(monkeypatch) -> None:
    tool = TransformObjectTool()
    cube = SimpleNamespace(
        name="Cube",
        location=(1.0, 2.0, 3.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    class FakeObjects(dict):
        def get(self, key: str, default=None):
            return super().get(key, default)

    fake_bpy = SimpleNamespace(
        data=SimpleNamespace(objects=FakeObjects({"Cube": cube})),
        context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)),
    )
    monkeypatch.setattr(transform_tools_module, "bpy", fake_bpy)

    result = tool.execute(
        context=None,
        params={
            "object_name": "Cube",
            "location": [10, 20, 30],
        },
    )

    assert cube.location == (10.0, 20.0, 30.0)
    assert result.outputs == {"object_name": "Cube"}
