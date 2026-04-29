from __future__ import annotations

from types import SimpleNamespace

from vectra.tools.spatial import world_bounds
from vectra.tools.spatial_constraints import (
    SpatialConstraintSolver,
    SpatialConstraint,
    constraints_for_tool_action,
    object_records_from_scene_objects,
    validate_spatial_records,
)


def _mesh(name: str, *, location: tuple[float, float, float], dimensions: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(name=name, type="MESH", location=location, dimensions=dimensions, bound_box=None, matrix_world=None)


def test_constraints_for_spatial_tool_action_are_typed() -> None:
    constraints = constraints_for_tool_action(
        "object.place_on_surface",
        {"target": "Lamp", "reference": "Table", "surface": "top"},
    )

    assert len(constraints) == 1
    assert constraints[0].kind == "support"
    assert constraints[0].target == "Lamp"
    assert constraints[0].reference == "Table"
    assert constraints[0].params == {"surface": "top", "offset": 0.0}


def test_constraint_solver_places_object_on_support_without_guessing_coordinates() -> None:
    table = _mesh("Table", location=(0.0, 0.0, 0.5), dimensions=(2.0, 2.0, 1.0))
    lamp = _mesh("Lamp", location=(3.0, 0.0, 0.5), dimensions=(0.4, 0.4, 1.0))
    solver = SpatialConstraintSolver([table, lamp])

    solution = solver.solve(SpatialConstraint(kind="support", target="Lamp", reference="Table"))

    assert solution.location == (0.0, 0.0, 1.5)


def test_constraint_solver_resolves_overlap_with_separating_axis() -> None:
    first = _mesh("First", location=(0.0, 0.0, 0.5), dimensions=(1.0, 1.0, 1.0))
    second = _mesh("Second", location=(0.25, 0.0, 0.5), dimensions=(1.0, 1.0, 1.0))
    solver = SpatialConstraintSolver([first, second])

    solution = solver.solve(SpatialConstraint(kind="no_overlap", target="Second", reference="First", params={"padding": 0.05}))

    assert solution.location[0] > second.location[0]
    second.location = solution.location
    assert min(world_bounds(first)["max"][0], world_bounds(second)["max"][0]) - max(world_bounds(first)["min"][0], world_bounds(second)["min"][0]) <= 0.0


def test_constraint_validation_reports_repair_actions_for_bad_geometry() -> None:
    floor = _mesh("Floor", location=(0.0, 0.0, 0.0), dimensions=(4.0, 4.0, 0.0))
    floating = _mesh("Floating", location=(0.0, 0.0, 3.0), dimensions=(1.0, 1.0, 1.0))

    report = validate_spatial_records(object_records_from_scene_objects([floor, floating]), affected_names=["Floating"])

    assert report.ok is False
    assert report.top_issues[0]["object"] == "Floating"
    assert any(action["tool"] == "object.snap_to_support" for action in report.repair_actions)
