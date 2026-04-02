from __future__ import annotations

from .models import AgentContext, ExecutionInstruction, ReasoningStep


def interpret_reasoning(reasoning: ReasoningStep, context: AgentContext) -> ExecutionInstruction:
    del context
    execution_metadata = reasoning.metadata if isinstance(reasoning.metadata, dict) else {}
    actions = execution_metadata.get("actions", [])
    if not isinstance(actions, list):
        actions = []

    if reasoning.status in {"complete", "clarify", "error"}:
        return ExecutionInstruction(
            kind="none",
            summary=reasoning.narration or reasoning.expected_outcome,
            expected_outcome=reasoning.expected_outcome,
            metadata=execution_metadata,
        )

    code = execution_metadata.get("code")
    if reasoning.preferred_execution_mode == "vectra-code" and isinstance(code, str) and code.strip():
        return ExecutionInstruction(
            kind="console_code",
            summary=reasoning.narration or "Prepared a bounded Blender code snippet.",
            code=code.strip(),
            expected_outcome=reasoning.expected_outcome,
            signature=code.strip(),
            metadata=execution_metadata,
        )

    return ExecutionInstruction(
        kind="tool_actions",
        summary=reasoning.narration or "Prepared the next tool action.",
        actions=[action for action in actions if isinstance(action, dict)],
        expected_outcome=reasoning.expected_outcome,
        signature=str(actions),
        metadata=execution_metadata,
    )
