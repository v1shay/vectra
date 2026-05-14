from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RelationType = Literal["contains", "supports", "above", "below", "near", "frames", "lights"]
TaskStatus = Literal["pending", "ready", "running", "complete", "blocked", "failed"]


@dataclass(frozen=True)
class WorldNode:
    id: str
    name: str
    kind: str
    location: tuple[float, float, float] | None = None
    dimensions: tuple[float, float, float] | None = None
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldRelation:
    source: str
    target: str
    relation: RelationType | str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldGraph:
    nodes: dict[str, WorldNode] = field(default_factory=dict)
    relations: list[WorldRelation] = field(default_factory=list)

    def nodes_by_role(self, role: str) -> list[WorldNode]:
        normalized = role.strip().lower()
        return [node for node in self.nodes.values() if (node.role or "").lower() == normalized]

    def names(self) -> set[str]:
        return {node.name for node in self.nodes.values()}


@dataclass(frozen=True)
class Affordance:
    name: str
    description: str = ""


@dataclass(frozen=True)
class IdentityLink:
    source_id: str
    target_id: str
    reason: str


@dataclass(frozen=True)
class SemanticNode:
    id: str
    label: str
    role: str
    affordances: list[Affordance] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticGraph:
    nodes: dict[str, SemanticNode] = field(default_factory=dict)
    identity_links: list[IdentityLink] = field(default_factory=list)


@dataclass(frozen=True)
class Dependency:
    before: str
    after: str
    reason: str


@dataclass(frozen=True)
class Blocker:
    id: str
    task_id: str
    reason: str
    severity: Literal["info", "warning", "error"] = "error"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentTask:
    id: str
    description: str
    obligation: str
    status: TaskStatus = "pending"
    dependencies: list[str] = field(default_factory=list)
    expected_roles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntentGoal:
    id: str
    prompt: str
    benchmark: str
    tasks: list[IntentTask] = field(default_factory=list)


@dataclass(frozen=True)
class IntentGraph:
    goal: IntentGoal
    dependencies: list[Dependency] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)

    def missing_obligations(self, completed_task_ids: set[str]) -> list[str]:
        return [
            task.obligation
            for task in self.goal.tasks
            if task.id not in completed_task_ids and task.status != "blocked"
        ]


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _role_from_record(record: dict[str, Any]) -> str | None:
    for key in ("semantic_role", "role", "vectra_role"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    custom = record.get("custom_properties")
    if isinstance(custom, dict):
        value = custom.get("vectra_role") or custom.get("semantic_role")
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = str(record.get("name", "")).lower()
    role_markers = {
        "floor": "floor",
        "light": "light",
        "camera": "camera",
    }
    for marker, role in role_markers.items():
        if marker in name:
            return role
    return None


def world_graph_from_scene_state(scene_state: dict[str, Any] | None) -> WorldGraph:
    if not isinstance(scene_state, dict):
        return WorldGraph()
    raw_objects = scene_state.get("objects", [])
    if not isinstance(raw_objects, list):
        return WorldGraph()

    nodes: dict[str, WorldNode] = {}
    for index, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, dict):
            continue
        name = raw_object.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        node_id = str(raw_object.get("id") or name)
        nodes[node_id] = WorldNode(
            id=node_id,
            name=name,
            kind=str(raw_object.get("type") or raw_object.get("kind") or "OBJECT"),
            location=_tuple3(raw_object.get("location")),
            dimensions=_tuple3(raw_object.get("dimensions") or raw_object.get("scale")),
            role=_role_from_record(raw_object),
            metadata={"index": index, **raw_object},
        )

    relations: list[WorldRelation] = []
    for source in nodes.values():
        for target in nodes.values():
            if source.id == target.id or source.location is None or target.location is None:
                continue
            if source.location[2] > target.location[2] + 0.2:
                relations.append(WorldRelation(source=source.id, target=target.id, relation="above"))
            if source.location[2] + 0.2 < target.location[2]:
                relations.append(WorldRelation(source=source.id, target=target.id, relation="below"))
    return WorldGraph(nodes=nodes, relations=relations)
