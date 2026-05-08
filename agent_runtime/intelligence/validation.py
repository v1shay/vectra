from __future__ import annotations

from dataclasses import dataclass, field

from .graphs import WorldGraph


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)


def validate_maintenance_bay_graph(world_graph: WorldGraph) -> ValidationResult:
    passed: list[str] = []
    failures: list[str] = []

    floors = world_graph.nodes_by_role("floor")
    horizontal_floor = False
    for floor in floors:
        dims = floor.dimensions
        if dims is not None and dims[2] <= min(dims[0], dims[1]) * 0.2:
            horizontal_floor = True
            break
    if horizontal_floor:
        passed.append("floor_z_up_horizontal")
    else:
        failures.append("missing horizontal Z-up floor")

    catwalks = world_graph.nodes_by_role("catwalk")
    raised_catwalk = any(node.location is not None and node.location[2] >= 1.5 for node in catwalks)
    if raised_catwalk:
        passed.append("raised_catwalk")
    else:
        failures.append("missing raised catwalk")

    workstations = world_graph.nodes_by_role("workstation")
    if len(workstations) >= 3:
        passed.append("three_workstations")
    else:
        failures.append(f"expected at least 3 workstations, found {len(workstations)}")

    for role, label in (
        ("cable", "cable bundle"),
        ("hazard_stripe", "hazard stripes"),
        ("overhead_light", "overhead lights"),
        ("camera", "camera framing"),
    ):
        if world_graph.nodes_by_role(role):
            passed.append(role)
        else:
            failures.append(f"missing {label}")

    return ValidationResult(ok=not failures, passed=passed, failures=failures)
