from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class TaskCreateRequest(BaseModel):
    prompt: str
    scene_state: dict[str, Any] = Field(default_factory=dict)
    images: list[Any] = Field(default_factory=list)


class TaskCreateResponse(BaseModel):
    status: str = "ok"
    message: str = "received"
    actions: list[Any] = Field(default_factory=list)
