from __future__ import annotations

from types import SimpleNamespace

import vectra.tools.mesh_tools as mesh_tools_module
import vectra.tools.transform_tools as transform_tools_module
from vectra.tools.mesh_tools import CreatePrimitiveTool
from vectra.tools.transform_tools import TransformObjectTool


def test_create_primitive_accepts_null_optional_fields() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params(
        {
            "primitive_type": "uv_sphere",
            "location": None,
            "name": None,
        }
    )

    assert validated == {
        "primitive_type": "uv_sphere",
        "name": None,
        "location": (0.0, 0.0, 0.0),
    }


def test_create_primitive_accepts_axis_mapping_and_blank_name() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params(
        {
            "primitive_type": "plane",
            "location": {"x": 0, "y": 0, "z": 0},
            "name": "",
        }
    )

    assert validated == {
        "primitive_type": "plane",
        "name": None,
        "location": (0.0, 0.0, 0.0),
    }


def test_create_primitive_accepts_partial_axis_mapping_with_defaults() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params(
        {
            "primitive_type": "uv_sphere",
            "location": {"x": 10},
        }
    )

    assert validated == {
        "primitive_type": "uv_sphere",
        "name": None,
        "location": (10.0, 0.0, 0.0),
    }


def test_create_primitive_accepts_empty_location_mapping_as_default_origin() -> None:
    tool = CreatePrimitiveTool()

    validated = tool.validate_params(
        {
            "primitive_type": "plane",
            "location": {},
            "name": "",
        }
    )

    assert validated == {
        "primitive_type": "plane",
        "name": None,
        "location": (0.0, 0.0, 0.0),
    }


def test_create_primitive_execute_handles_llm_style_optional_fields(monkeypatch) -> None:
    tool = CreatePrimitiveTool()
    context = SimpleNamespace(active_object=None)
    calls: dict[str, object] = {}

    def primitive_uv_sphere_add(*, location):
        calls["location"] = location
        context.active_object = SimpleNamespace(name="Sphere")
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(
            mesh=SimpleNamespace(primitive_uv_sphere_add=primitive_uv_sphere_add)
        )
    )
    monkeypatch.setattr(mesh_tools_module, "bpy", fake_bpy)

    result = tool.execute(
        context,
        {
            "primitive_type": "uv_sphere",
            "location": {},
            "name": "",
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


def test_transform_validate_accepts_partial_axis_mapping() -> None:
    tool = TransformObjectTool()

    validated = tool.validate_params(
        {
            "object_name": "Cube",
            "location": {"x": 10},
        }
    )

    assert validated == {
        "object_name": "Cube",
        "location": {"x": 10.0},
    }


def test_transform_execute_preserves_unspecified_axes(monkeypatch) -> None:
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
            "location": {"x": 10},
        },
    )

    assert cube.location == (10.0, 2.0, 3.0)
    assert result.outputs == {"object_name": "Cube"}
