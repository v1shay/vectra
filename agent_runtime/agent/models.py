from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ExecutionMode = Literal["vectra-dev", "vectra-code"]
ReasoningStatus = Literal["ok", "clarify", "complete", "error"]
ExecutionKind = Literal["tool_actions", "console_code", "none"]


@dataclass(frozen=True)
class AgentContext:
    user_prompt: str
    scene_state: dict[str, Any]
    screenshot: dict[str, Any] | None
    history: list[dict[str, Any]]
    iteration: int
    execution_mode: ExecutionMode
    memory_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReasoningStep:
    status: ReasoningStatus
    narration: str
    understanding: str
    plan: list[str]
    intended_actions: list[str]
    expected_outcome: str
    preferred_execution_mode: ExecutionMode
    continue_loop: bool
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    question: str | None = None
    error: str | None = None
    uncertainty_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionInstruction:
    kind: ExecutionKind
    summary: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    code: str | None = None
    expected_outcome: str = ""
    signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
