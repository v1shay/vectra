from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class TaskCreateRequest(BaseModel):
    prompt: str
    scene_state: dict[str, Any] = Field(default_factory=dict)
    images: list[Any] = Field(default_factory=list)


class ActionModel(BaseModel):
    action_id: str | None = None
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class TaskCreateResponse(BaseModel):
    status: str = "ok"
    message: str = "planned"
    actions: list[ActionModel] = Field(default_factory=list)
