from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ExecutionMode = Literal["vectra-dev", "vectra-code"]
DirectorStatus = Literal["ok", "clarify", "complete", "error"]
ComplexityLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ControllerDecision:
    needs_scene_context: bool = True
    needs_visual_feedback: bool = False
    complexity: ComplexityLevel = "medium"
    provider: str | None = None
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssumptionRecord:
    key: str
    value: Any
    reason: str


@dataclass(frozen=True)
class ObservationSummary:
    summary: str
    created_objects: list[str] = field(default_factory=list)
    removed_objects: list[str] = field(default_factory=list)
    moved_objects: list[str] = field(default_factory=list)
    changed_objects: list[str] = field(default_factory=list)
    screenshot_available: bool = False
    meaningful_change: bool = False


@dataclass(frozen=True)
class BudgetState:
    complexity: ComplexityLevel
    turn_budget: int
    turns_used: int
    turns_remaining: int
    completion_mode_active: bool
    core_task_started: bool


@dataclass(frozen=True)
class LoopMemoryRecord:
    prompt: str
    summary: str
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observation_summary: str = ""
    category: str = "director_loop"


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    model: str
    assistant_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    provider_chain: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DirectorContext:
    user_prompt: str
    scene_state: dict[str, Any]
    screenshot: dict[str, Any] | None
    history: list[dict[str, Any]]
    iteration: int
    execution_mode: ExecutionMode
    memory_results: list[dict[str, Any]] = field(default_factory=list)
    latest_observation: ObservationSummary | None = None
    budget_state: BudgetState | None = None


@dataclass(frozen=True)
class DirectorTurn:
    status: DirectorStatus
    message: str
    narration: str
    understanding: str
    plan: list[str]
    intended_actions: list[str]
    expected_outcome: str
    continue_loop: bool
    assumptions: list[AssumptionRecord] = field(default_factory=list)
    question: str | None = None
    error: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    code: str | None = None
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
