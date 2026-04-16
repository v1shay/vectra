from __future__ import annotations

from types import SimpleNamespace

import pytest

import vectra.tools.floor_tools as floor_tools_module
import vectra.tools.spatial_tools as spatial_tools_module
from vectra.tools.base import ToolValidationError
from vectra.tools.floor_tools import EnsureFloorTool
from vectra.tools.registry import ToolRegistry
from vectra.tools.spatial import (
    all_face_centers,
    floor_contact_record,
    is_wall_like,
    spatial_anchors,
    spatial_metadata_for_object,
    spatial_relations,
    face_center,
    lowest_z,
    world_bounds,
)
from vectra.tools.spatial_tools import PlaceAgainstTool, PlaceOnSurfaceTool, PlaceRelativeTool


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


def test_spatial_metadata_relations_and_anchors_are_deterministic() -> None:
    floor = {
        "name": "Floor",
        "type": "MESH",
        "bounds": {"min": [-4.0, -4.0, 0.0], "max": [4.0, 4.0, 0.0]},
    }
    wall = {
        "name": "BackWall",
        "type": "MESH",
        "bounds": {"min": [-4.0, -4.0, 0.0], "max": [4.0, -3.8, 3.0]},
    }
    bed = {
        "name": "Bed",
        "type": "MESH",
        "bounds": {"min": [-1.0, -3.8, 0.0], "max": [1.0, -2.0, 0.8]},
    }
    nightstand = {
        "name": "Nightstand",
        "type": "MESH",
        "bounds": {"min": [1.0, -3.4, 0.0], "max": [1.8, -2.6, 0.8]},
    }
    lamp = {
        "name": "Lamp",
        "type": "MESH",
        "bounds": {"min": [1.2, -3.2, 0.8], "max": [1.6, -2.8, 1.6]},
    }
    objects = [bed, floor, lamp, nightstand, wall]

    bed_metadata = spatial_metadata_for_object(bed, floor_candidates=[floor])
    relation_records = spatial_relations(objects)
    anchor_records = spatial_anchors(objects)

    assert bed_metadata["center"] == [0.0, -2.9, 0.4]
    assert bed_metadata["half_extents"] == [1.0, 0.8999999999999999, 0.4]
    assert bed_metadata["grounded"] is True
    assert bed_metadata["floor_contact"] == {"object": "Floor", "gap": 0.0}
    assert is_wall_like(wall) is True
    assert floor_contact_record(lamp, [nightstand]) == {"object": "Nightstand", "gap": 0.0}
    assert {"source": "Bed", "target": "BackWall", "relation": "against"} in relation_records
    assert {"source": "Nightstand", "target": "Bed", "relation": "next_to"} in relation_records
    assert {"source": "Lamp", "target": "Nightstand", "relation": "on"} in relation_records
    assert relation_records == spatial_relations(list(reversed(objects)))
    assert anchor_records == spatial_anchors(list(reversed(objects)))
    assert "top" in all_face_centers(world_bounds(bed))
    assert any(anchor["name"] == "BackWall.wall.inner" for anchor in anchor_records)
    assert any(anchor["type"] == "scene_corner" for anchor in anchor_records)


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


def test_place_against_preserves_grounded_axes_for_bed_against_wall(monkeypatch) -> None:
    bed = _mesh("Bed", location=(0.0, 0.0, 0.5), dimensions=(2.0, 2.0, 1.0))
    wall = _mesh("BackWall", location=(0.0, -3.0, 1.5), dimensions=(6.0, 0.2, 3.0))

    monkeypatch.setattr(spatial_tools_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(spatial_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(
        spatial_tools_module,
        "resolve_object",
        lambda context, name: {"Bed": bed, "BackWall": wall}.get(name),
    )

    result = PlaceAgainstTool().execute(
        context=SimpleNamespace(),
        params={"target": "Bed", "reference": "BackWall", "side": "front", "offset": 0.0},
    )

    assert bed.location == (0.0, -1.9, 0.5)
    assert result.outputs["placement_mode"] == "against_contact"
    assert world_bounds(bed)["min"][1] == pytest.approx(world_bounds(wall)["max"][1])
    assert world_bounds(bed)["min"][2] == pytest.approx(0.0)


def test_place_relative_keeps_beside_object_grounded(monkeypatch) -> None:
    bed = _mesh("Bed", location=(0.0, -1.9, 0.5), dimensions=(2.0, 2.0, 1.0))
    nightstand = _mesh("Nightstand", location=(0.0, -1.9, 0.4), dimensions=(0.6, 0.8, 0.8))

    monkeypatch.setattr(spatial_tools_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(spatial_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(
        spatial_tools_module,
        "resolve_object",
        lambda context, name: {"Bed": bed, "Nightstand": nightstand}.get(name),
    )

    PlaceRelativeTool().execute(
        context=SimpleNamespace(),
        params={"target": "Nightstand", "reference": "Bed", "relation": "right_of", "distance": 0.2},
    )

    assert nightstand.location == (1.5, -1.9, 0.4)
    assert world_bounds(nightstand)["min"][0] == pytest.approx(world_bounds(bed)["max"][0] + 0.2)
    assert world_bounds(nightstand)["min"][2] == pytest.approx(0.0)


def test_place_on_surface_puts_lamp_on_nightstand(monkeypatch) -> None:
    nightstand = _mesh("Nightstand", location=(1.5, -1.9, 0.4), dimensions=(0.6, 0.8, 0.8))
    lamp = _mesh("Lamp", location=(0.0, 0.0, 0.3), dimensions=(0.3, 0.3, 0.6))

    monkeypatch.setattr(spatial_tools_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(spatial_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(
        spatial_tools_module,
        "resolve_object",
        lambda context, name: {"Lamp": lamp, "Nightstand": nightstand}.get(name),
    )

    PlaceOnSurfaceTool().execute(
        context=SimpleNamespace(),
        params={"target": "Lamp", "reference": "Nightstand", "surface": "top"},
    )

    assert lamp.location == (1.5, -1.9, 1.1)
    assert world_bounds(lamp)["min"][2] == pytest.approx(world_bounds(nightstand)["max"][2])


def test_place_against_can_compose_corner_placement_without_scene_template(monkeypatch) -> None:
    left_wall = _mesh("LeftWall", location=(-3.0, 0.0, 1.5), dimensions=(0.2, 6.0, 3.0))
    back_wall = _mesh("BackWall", location=(0.0, -3.0, 1.5), dimensions=(6.0, 0.2, 3.0))
    plant = _mesh("Plant", location=(-2.0, -2.0, 0.5), dimensions=(0.5, 0.5, 1.0))
    objects = {"LeftWall": left_wall, "BackWall": back_wall, "Plant": plant}

    monkeypatch.setattr(spatial_tools_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(spatial_tools_module, "ensure_object_mode", lambda context: None)
    monkeypatch.setattr(spatial_tools_module, "resolve_object", lambda context, name: objects.get(name))

    tool = PlaceAgainstTool()
    tool.execute(context=SimpleNamespace(), params={"target": "Plant", "reference": "LeftWall", "side": "right"})
    tool.execute(context=SimpleNamespace(), params={"target": "Plant", "reference": "BackWall", "side": "front"})

    assert plant.location == (-2.65, -2.65, 0.5)
    assert world_bounds(plant)["min"][0] == pytest.approx(world_bounds(left_wall)["max"][0])
    assert world_bounds(plant)["min"][1] == pytest.approx(world_bounds(back_wall)["max"][1])
    assert world_bounds(plant)["min"][2] == pytest.approx(0.0)


def test_spatial_tools_reject_invalid_relation_parameters() -> None:
    with pytest.raises(ToolValidationError, match="'distance' must be finite"):
        PlaceRelativeTool().validate_params(
            {"target": "A", "reference": "B", "relation": "right_of", "distance": float("nan")}
        )
    with pytest.raises(ToolValidationError, match="'offset' must be greater than or equal to 0"):
        PlaceAgainstTool().validate_params({"target": "A", "reference": "B", "side": "front", "offset": -0.1})
    with pytest.raises(ToolValidationError, match="'target' must be a non-empty string"):
        PlaceOnSurfaceTool().validate_params({"target": None, "reference": "B", "surface": "top"})


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
