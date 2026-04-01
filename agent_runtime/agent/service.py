from __future__ import annotations

from typing import Any

from agent_runtime.models import (
    AgentStepRequest,
    AgentStepResponse,
    ExecutionPayloadModel,
    HistoryEntryModel,
    ScreenshotModel,
)
from agent_runtime.utils import model_to_dict
from agent_runtime.memory import MemoryManager

from .interpreter import interpret_reasoning
from .models import AgentContext
from .reasoner import reason_step


class AgentService:
    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory_manager = memory_manager or MemoryManager()

    def _context_from_request(self, request: AgentStepRequest) -> AgentContext:
        prompt = request.prompt.strip()
        memory_results = self.memory_manager.query_memory(prompt, top_k=5)
        return AgentContext(
            user_prompt=prompt,
            scene_state=model_to_dict(request).get("scene_state", {}),
            screenshot=model_to_dict(request.screenshot) if request.screenshot is not None else None,
            history=[model_to_dict(entry) for entry in request.history],
            iteration=max(int(request.iteration), 1),
            execution_mode=request.execution_mode,
            memory_results=memory_results,
        )

    def step(self, request: AgentStepRequest) -> AgentStepResponse:
        context = self._context_from_request(request)
        reasoning = reason_step(context)
        instruction = interpret_reasoning(reasoning, context)
        return AgentStepResponse(
            status=reasoning.status,
            message=reasoning.narration or reasoning.expected_outcome,
            narration=reasoning.narration,
            understanding=reasoning.understanding,
            plan=list(reasoning.plan),
            intended_actions=list(reasoning.intended_actions),
            uncertainty_notes=list(reasoning.uncertainty_notes),
            preferred_execution_mode=reasoning.preferred_execution_mode,
            continue_loop=reasoning.continue_loop,
            question=reasoning.question,
            expected_outcome=reasoning.expected_outcome,
            execution=ExecutionPayloadModel(
                kind=instruction.kind,
                summary=instruction.summary,
                actions=instruction.actions,
                code=instruction.code,
                expected_outcome=instruction.expected_outcome,
            ),
        )
