from __future__ import annotations

from typing import Any, Literal

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


class AssumptionModel(BaseModel):
    key: str
    value: Any = None
    reason: str = ""


class TaskCreateResponse(BaseModel):
    status: str = "ok"
    message: str = "planned"
    actions: list[ActionModel] = Field(default_factory=list)
    assumptions: list[AssumptionModel] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


ExecutionMode = Literal["vectra-dev", "vectra-code"]
AgentStepStatus = Literal["ok", "clarify", "complete", "error"]
ExecutionPayloadKind = Literal["tool_actions", "console_code", "none"]


class ScreenshotModel(BaseModel):
    available: bool = False
    path: str | None = None
    reason: str | None = None


class HistoryEntryModel(BaseModel):
    iteration: int = 0
    role: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentStepRequest(BaseModel):
    prompt: str
    scene_state: dict[str, Any] = Field(default_factory=dict)
    screenshot: ScreenshotModel | None = None
    history: list[HistoryEntryModel] = Field(default_factory=list)
    iteration: int = 1
    execution_mode: ExecutionMode = "vectra-dev"


class ExecutionPayloadModel(BaseModel):
    kind: ExecutionPayloadKind = "none"
    summary: str = ""
    actions: list[ActionModel] = Field(default_factory=list)
    code: str | None = None
    expected_outcome: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStepResponse(BaseModel):
    status: AgentStepStatus = "ok"
    message: str = ""
    narration: str = ""
    understanding: str = ""
    plan: list[str] = Field(default_factory=list)
    intended_actions: list[str] = Field(default_factory=list)
    assumptions: list[AssumptionModel] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    preferred_execution_mode: ExecutionMode = "vectra-dev"
    continue_loop: bool = False
    question: str | None = None
    error: str | None = None
    expected_outcome: str = ""
    execution: ExecutionPayloadModel = Field(default_factory=ExecutionPayloadModel)
