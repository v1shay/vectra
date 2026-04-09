from __future__ import annotations

from types import SimpleNamespace

import vectra.tools.floor_tools as floor_tools_module
import vectra.tools.spatial_tools as spatial_tools_module
from vectra.tools.floor_tools import EnsureFloorTool
from vectra.tools.registry import ToolRegistry
from vectra.tools.spatial import face_center, lowest_z, world_bounds
from vectra.tools.spatial_tools import PlaceOnSurfaceTool


def _mesh(name: str, *, location: tuple[float, float, float], dimensions: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        type="MESH",
        location=location,
        dimensions=dimensions,
        bound_box=None,
        matrix_world=None,
        scale=None,
    )


class _CornerVector:
    def __init__(self, coords: tuple[float, float, float]) -> None:
        self._coords = coords

    def __iter__(self):
        return iter(self._coords)


class _IdentityMatrix:
    def __matmul__(self, other):
        return tuple(other)


def test_world_bounds_face_center_and_lowest_z_are_deterministic() -> None:
    cube = {
        "name": "Cube",
        "type": "MESH",
        "location": [2.0, 3.0, 1.0],
        "dimensions": [4.0, 2.0, 2.0],
    }

    bounds = world_bounds(cube)

    assert bounds == {
        "min": (0.0, 2.0, 0.0),
        "max": (4.0, 4.0, 2.0),
    }
    assert face_center(bounds, "top") == (2.0, 3.0, 2.0)
    assert lowest_z(cube) == 0.0


def test_world_bounds_accepts_iterable_blender_style_bound_box_corners() -> None:
    obj = SimpleNamespace(
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
        matrix_world=_IdentityMatrix(),
        location=(0.0, 0.0, 0.0),
        dimensions=(2.0, 2.0, 2.0),
        type="MESH",
    )

    assert world_bounds(obj) == {
        "min": (-1.0, -1.0, -1.0),
        "max": (1.0, 1.0, 1.0),
    }


def test_tool_registry_discovers_spatial_phase_a_tools() -> None:
    registry = ToolRegistry()
    registry.discover()

    discovered_tools = registry.list_tools()

    assert "object.place_on_surface" in discovered_tools
    assert "object.place_against" in discovered_tools
    assert "object.place_relative" in discovered_tools
    assert "object.align_to" in discovered_tools
    assert "scene.ensure_floor" in discovered_tools


def test_place_on_surface_tool_grounds_target_on_reference_top(monkeypatch) -> None:
    target = _mesh("Target", location=(0.0, 0.0, 0.0), dimensions=(2.0, 2.0, 2.0))
    reference = _mesh("Floor", location=(0.0, 0.0, 0.0), dimensions=(8.0, 8.0, 0.0))

    monkeypatch.setattr(spatial_tools_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(spatial_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(
        spatial_tools_module,
        "resolve_object",
        lambda context, name: {"Target": target, "Floor": reference}.get(name),
    )

    result = PlaceOnSurfaceTool().execute(
        context=SimpleNamespace(),
        params={"target": "Target", "reference": "Floor", "surface": "top", "offset": 0.0},
    )

    assert target.location == (0.0, 0.0, 1.0)
    assert result.outputs["placement_mode"] == "surface_contact"
    assert result.outputs["reference_object"] == "Floor"


def test_ensure_floor_normalizes_existing_floor_candidate(monkeypatch) -> None:
    floor = _mesh("ExistingFloor", location=(0.0, 0.0, -1.0), dimensions=(8.0, 8.0, 0.0))

    monkeypatch.setattr(floor_tools_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(floor_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(floor_tools_module, "scene_objects", lambda context: [floor])

    result = EnsureFloorTool().execute(context=SimpleNamespace(), params={})

    assert floor.location == (0.0, 0.0, 0.0)
    assert result.outputs["normalized_existing"] is True
    assert result.outputs["created_new"] is False


def test_ensure_floor_creates_floor_when_scene_has_no_mesh_objects(monkeypatch) -> None:
    created_floor = _mesh("Plane", location=(0.0, 0.0, 0.0), dimensions=(2.0, 2.0, 0.0))
    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(mesh=SimpleNamespace()),
        context=SimpleNamespace(active_object=None),
    )

    def primitive_plane_add(*, location):
        assert location == (0.0, 0.0, 0.0)
        fake_bpy.context.active_object = created_floor
        return {"FINISHED"}

    fake_bpy.ops.mesh.primitive_plane_add = primitive_plane_add

    monkeypatch.setattr(floor_tools_module, "bpy", fake_bpy)
    monkeypatch.setattr(floor_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(floor_tools_module, "scene_objects", lambda context: [])

    result = EnsureFloorTool().execute(context=SimpleNamespace(), params={})

    assert created_floor.name == "Floor"
    assert created_floor.scale == (4.0, 4.0, 1.0)
    assert result.outputs["created_new"] is True
    assert result.outputs["normalized_existing"] is False
