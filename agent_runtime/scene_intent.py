from __future__ import annotations

import ast
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field


SceneIntentStatus = Literal["ok", "no_action"]
EntitySource = Literal["create", "reference", "auto"]
RelationshipKind = Literal[
    "left_of",
    "right_of",
    "above",
    "below",
    "next_to",
    "relative_offset",
]
ConstructionStepKind = Literal[
    "ensure_entity",
    "apply_transform",
    "resolve_group",
    "satisfy_relation",
]

_SUPPORTED_ENTITY_KINDS = {
    "box": "cube",
    "cube": "cube",
    "plane": "plane",
    "sphere": "uv_sphere",
    "uv sphere": "uv_sphere",
    "uv_sphere": "uv_sphere",
}
_SECTION_HEADERS = {
    "STATUS",
    "CONFIDENCE",
    "REASONING",
    "ENTITIES",
    "RELATIONSHIPS",
    "GROUPS",
    "CONSTRUCTION_STEPS",
    "UNCERTAINTY",
    "METADATA",
}


class SceneTransformIntent(BaseModel):
    offset: list[float] | None = None
    rotation_degrees: list[float] | None = None
    scale: list[float] | None = None


class SceneEntity(BaseModel):
    logical_id: str
    kind: str
    display_name: str | None = None
    quantity: int = Field(default=1, ge=1)
    source: EntitySource = "create"
    reference_name: str | None = None
    group_id: str | None = None
    initial_transform: SceneTransformIntent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneRelationship(BaseModel):
    logical_id: str
    relation_type: RelationshipKind
    source_id: str
    target_id: str | None = None
    target_group_id: str | None = None
    offset: list[float] | None = None
    distance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneGroup(BaseModel):
    logical_id: str
    entity_ids: list[str] = Field(default_factory=list)
    pronouns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstructionStep(BaseModel):
    logical_id: str
    kind: ConstructionStepKind
    entity_id: str | None = None
    relationship_id: str | None = None
    group_id: str | None = None
    offset: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneIntent(BaseModel):
    status: SceneIntentStatus = "no_action"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    entities: list[SceneEntity] = Field(default_factory=list)
    relationships: list[SceneRelationship] = Field(default_factory=list)
    groups: list[SceneGroup] = Field(default_factory=list)
    construction_steps: list[ConstructionStep] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneIntentParseError(Exception):
    """Raised when the LLM output cannot be parsed into a SceneIntent."""


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    return normalized or None


def _normalize_kind(raw_kind: str) -> str:
    normalized = (_normalize_text(raw_kind) or "").lower()
    return _SUPPORTED_ENTITY_KINDS.get(normalized, normalized)


def _sanitize_identifier(value: str, *, default: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    return cleaned or default


def _coerce_vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise SceneIntentParseError("Expected a 3-item vector")
    return [float(component) for component in value]


def _coerce_transform(raw_transform: Any) -> SceneTransformIntent | None:
    if raw_transform is None:
        return None
    if isinstance(raw_transform, SceneTransformIntent):
        return raw_transform
    if not isinstance(raw_transform, dict):
        raise SceneIntentParseError("Entity transform must be an object")

    return SceneTransformIntent(
        offset=_coerce_vector3(raw_transform.get("offset")),
        rotation_degrees=_coerce_vector3(raw_transform.get("rotation_degrees")),
        scale=_coerce_vector3(raw_transform.get("scale")),
    )


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    if value.startswith(("[", "{", "(", "'")) or value.startswith('"'):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _parse_bullet_mapping(line: str) -> dict[str, Any]:
    raw_line = line.strip()
    if raw_line.startswith("-"):
        raw_line = raw_line[1:].strip()

    result: dict[str, Any] = {}
    for chunk in raw_line.split(";"):
        if ":" not in chunk:
            continue
        key, raw_value = chunk.split(":", 1)
        result[_sanitize_identifier(key, default="field")] = _parse_scalar(raw_value)
    return result


def _parse_sectioned_text(content: str) -> dict[str, Any]:
    sections: dict[str, list[str]] = {}
    current_header: str | None = None
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if ":" in stripped:
            header_candidate = stripped.split(":", 1)[0].strip().upper()
            if header_candidate in _SECTION_HEADERS:
                current_header = header_candidate
                remainder = stripped.split(":", 1)[1].strip()
                sections.setdefault(current_header, [])
                if remainder:
                    sections[current_header].append(remainder)
                continue

        if current_header is not None:
            sections.setdefault(current_header, []).append(stripped)

    payload: dict[str, Any] = {
        "status": "no_action",
        "confidence": 0.0,
        "reasoning": "",
        "entities": [],
        "relationships": [],
        "groups": [],
        "construction_steps": [],
        "uncertainty_notes": [],
        "metadata": {},
    }

    if "STATUS" in sections and sections["STATUS"]:
        payload["status"] = str(_parse_scalar(sections["STATUS"][0])).strip().lower()
    if "CONFIDENCE" in sections and sections["CONFIDENCE"]:
        payload["confidence"] = float(_parse_scalar(sections["CONFIDENCE"][0]))
    if "REASONING" in sections and sections["REASONING"]:
        payload["reasoning"] = " ".join(sections["REASONING"]).strip()

    for key, target in (
        ("ENTITIES", "entities"),
        ("RELATIONSHIPS", "relationships"),
        ("GROUPS", "groups"),
        ("CONSTRUCTION_STEPS", "construction_steps"),
    ):
        for line in sections.get(key, []):
            if not line.startswith("-"):
                continue
            payload[target].append(_parse_bullet_mapping(line))

    for line in sections.get("UNCERTAINTY", []):
        if line.startswith("-"):
            payload["uncertainty_notes"].append(line[1:].strip())
        else:
            payload["uncertainty_notes"].append(line.strip())

    for line in sections.get("METADATA", []):
        if line.startswith("-"):
            entry = _parse_bullet_mapping(line)
            payload["metadata"].update(entry)
        elif ":" in line:
            key, raw_value = line.split(":", 1)
            payload["metadata"][_sanitize_identifier(key, default="key")] = _parse_scalar(raw_value)

    return payload


def parse_scene_intent_content(content: str) -> SceneIntent:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = _parse_sectioned_text(content)

    if not isinstance(parsed, dict):
        raise SceneIntentParseError("Scene intent response must be a JSON object or labeled object")

    try:
        return SceneIntent.model_validate(parsed)
    except Exception as exc:  # pragma: no cover - pydantic already gives the details
        raise SceneIntentParseError(f"Scene intent response did not match schema: {exc}") from exc


def normalize_scene_intent(
    intent: SceneIntent,
    *,
    minimum_confidence: float,
) -> SceneIntent:
    if intent.status != "ok":
        return SceneIntent(
            status="no_action",
            confidence=float(intent.confidence),
            reasoning=intent.reasoning,
            uncertainty_notes=list(intent.uncertainty_notes),
            metadata=dict(intent.metadata),
        )

    if float(intent.confidence) < minimum_confidence:
        return SceneIntent(
            status="no_action",
            confidence=float(intent.confidence),
            reasoning="Intent confidence below threshold",
            uncertainty_notes=list(intent.uncertainty_notes),
            metadata=dict(intent.metadata),
        )

    normalized_entities: list[SceneEntity] = []
    for index, entity in enumerate(intent.entities, start=1):
        logical_id = _sanitize_identifier(entity.logical_id, default=f"entity_{index}")
        kind = _normalize_kind(entity.kind)
        if kind not in set(_SUPPORTED_ENTITY_KINDS.values()):
            return SceneIntent(
                status="no_action",
                confidence=float(intent.confidence),
                reasoning=f"Unsupported entity kind '{entity.kind}'",
                uncertainty_notes=list(intent.uncertainty_notes),
                metadata=dict(intent.metadata),
            )

        group_id = _sanitize_identifier(entity.group_id, default="") if entity.group_id else None
        normalized_entities.append(
            entity.model_copy(
                update={
                    "logical_id": logical_id,
                    "kind": kind,
                    "display_name": _normalize_text(entity.display_name),
                    "reference_name": _normalize_text(entity.reference_name),
                    "group_id": group_id,
                    "initial_transform": _coerce_transform(
                        entity.initial_transform.model_dump()
                        if isinstance(entity.initial_transform, SceneTransformIntent)
                        else entity.initial_transform
                    ),
                }
            )
        )

    normalized_relationships: list[SceneRelationship] = []
    for index, relationship in enumerate(intent.relationships, start=1):
        logical_id = _sanitize_identifier(relationship.logical_id, default=f"relationship_{index}")
        normalized_relationships.append(
            relationship.model_copy(
                update={
                    "logical_id": logical_id,
                    "source_id": _sanitize_identifier(relationship.source_id, default="source"),
                    "target_id": (
                        _sanitize_identifier(relationship.target_id, default="target")
                        if relationship.target_id
                        else None
                    ),
                    "target_group_id": (
                        _sanitize_identifier(relationship.target_group_id, default="group")
                        if relationship.target_group_id
                        else None
                    ),
                    "offset": _coerce_vector3(relationship.offset),
                }
            )
        )

    normalized_groups: list[SceneGroup] = []
    for index, group in enumerate(intent.groups, start=1):
        logical_id = _sanitize_identifier(group.logical_id, default=f"group_{index}")
        normalized_groups.append(
            group.model_copy(
                update={
                    "logical_id": logical_id,
                    "entity_ids": [
                        _sanitize_identifier(entity_id, default=f"entity_{position}")
                        for position, entity_id in enumerate(group.entity_ids, start=1)
                    ],
                    "pronouns": [(_normalize_text(pronoun) or pronoun).lower() for pronoun in group.pronouns],
                }
            )
        )

    normalized_steps: list[ConstructionStep] = []
    for index, step in enumerate(intent.construction_steps, start=1):
        normalized_steps.append(
            step.model_copy(
                update={
                    "logical_id": _sanitize_identifier(step.logical_id, default=f"step_{index}"),
                    "entity_id": (
                        _sanitize_identifier(step.entity_id, default=f"entity_{index}")
                        if step.entity_id
                        else None
                    ),
                    "relationship_id": (
                        _sanitize_identifier(step.relationship_id, default=f"relationship_{index}")
                        if step.relationship_id
                        else None
                    ),
                    "group_id": (
                        _sanitize_identifier(step.group_id, default=f"group_{index}")
                        if step.group_id
                        else None
                    ),
                    "offset": _coerce_vector3(step.offset),
                }
            )
        )

    entity_ids = {entity.logical_id for entity in normalized_entities}
    for relationship in normalized_relationships:
        if relationship.source_id not in entity_ids:
            return SceneIntent(
                status="no_action",
                confidence=float(intent.confidence),
                reasoning=f"Relationship source '{relationship.source_id}' was not declared as an entity",
                uncertainty_notes=list(intent.uncertainty_notes),
                metadata=dict(intent.metadata),
            )
        if relationship.target_id and relationship.target_id not in entity_ids:
            return SceneIntent(
                status="no_action",
                confidence=float(intent.confidence),
                reasoning=f"Relationship target '{relationship.target_id}' was not declared as an entity",
                uncertainty_notes=list(intent.uncertainty_notes),
                metadata=dict(intent.metadata),
            )

    if not normalized_entities and not normalized_steps:
        return SceneIntent(
            status="no_action",
            confidence=float(intent.confidence),
            reasoning="No scene entities were produced",
            uncertainty_notes=list(intent.uncertainty_notes),
            metadata=dict(intent.metadata),
        )

    return SceneIntent(
        status="ok",
        confidence=float(intent.confidence),
        reasoning=intent.reasoning,
        entities=normalized_entities,
        relationships=normalized_relationships,
        groups=normalized_groups,
        construction_steps=normalized_steps,
        uncertainty_notes=list(intent.uncertainty_notes),
        metadata=dict(intent.metadata),
    )
