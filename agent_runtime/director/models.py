from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ExecutionMode = Literal["vectra-dev", "vectra-code"]
DirectorStatus = Literal["ok", "clarify", "complete", "error"]
ComplexityLevel = Literal["low", "medium", "high"]
RuntimeState = Literal[
    "awaiting_model_response",
    "provider_transport_failure",
    "provider_deadline_exceeded",
    "tool_call_parse_failure",
    "tool_validation_failure",
    "no_action_response",
    "valid_action_batch_ready",
    "fallback_provider_invoked",
]
ProviderParseStatus = Literal["ok", "tool_call_parse_failure", "no_action_response"]


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
class ToolCallValidationIssue:
    tool_name: str
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
    progress_score: float = 0.0
    progress_reasons: list[str] = field(default_factory=list)
    visible_animation: bool = False


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
class ProviderAdapterCapabilities:
    structured_tools: bool = True
    embeds_tools_in_prompt: bool = False
    supports_parallel_tool_calls: bool = True


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    model: str
    transport: str
    runtime_state: RuntimeState
    response_type: str = "unknown"
    parsed_tool_call_count: int = 0
    failure_reason: str = ""
    request_metadata: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedProviderResponse:
    assistant_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    response_type: str = "unknown"
    parse_status: ProviderParseStatus = "ok"
    failure_reason: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    model: str
    transport: str = "responses"
    parsed: ParsedProviderResponse = field(default_factory=ParsedProviderResponse)
    provider_chain: list[str] = field(default_factory=list)
    attempts: list[ProviderAttempt] = field(default_factory=list)
    adapter_capabilities: ProviderAdapterCapabilities = field(default_factory=ProviderAdapterCapabilities)
    runtime_state: RuntimeState = "valid_action_batch_ready"

    @property
    def assistant_text(self) -> str:
        return self.parsed.assistant_text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.parsed.tool_calls

    @property
    def raw_response(self) -> dict[str, Any]:
        return self.parsed.raw_response

    @property
    def response_type(self) -> str:
        return self.parsed.response_type

    @property
    def parse_status(self) -> ProviderParseStatus:
        return self.parsed.parse_status

    @property
    def failure_reason(self) -> str:
        return self.parsed.failure_reason


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
