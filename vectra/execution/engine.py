from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from typing import Any

from .normalization import normalize_action_params
from ..tools.base import ToolExecutionError, ToolValidationError
from ..tools.registry import ToolRegistry, ToolRegistryError, get_default_registry
from ..utils.logging import (
    get_vectra_logger,
    log_action_failure,
    log_action_start,
    log_action_success,
    log_execution_report,
)


class ActionValidationError(Exception):
    """Raised when an action is structurally invalid."""


class ReferenceResolutionError(Exception):
    """Raised when an action reference cannot be resolved."""


@dataclass(frozen=True)
class NormalizedAction:
    action_id: str | None
    tool: str
    params: dict[str, Any]


@dataclass
class ActionExecutionResult:
    action_id: str | None
    tool: str
    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    repairs: list[str] = field(default_factory=list)


@dataclass
class ExecutionReport:
    success: bool
    results: list[ActionExecutionResult]
    failed_action_id: str | None = None
    message: str = ""
    repairs: list[str] = field(default_factory=list)


class ExecutionEngine:
    def __init__(self, registry: ToolRegistry | None = None, logger: Any = None) -> None:
        if registry is None:
            self.registry = get_default_registry()
            self.registry.discover()
        else:
            self.registry = registry
        self.logger = logger or get_vectra_logger("vectra.execution")
        self._transient_failures_seen: set[str] = set()

    def run(self, context: Any, actions: Sequence[dict[str, Any]]) -> ExecutionReport:
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes, bytearray)):
            raise ActionValidationError("Actions must be a sequence")

        results: list[ActionExecutionResult] = []
        outputs_by_action: dict[str, dict[str, Any]] = {}
        seen_action_ids: set[str] = set()
        report_repairs: list[str] = []

        for index, raw_action in enumerate(actions):
            action_id = self._extract_action_id(raw_action)
            tool_name = self._extract_tool_name(raw_action)

            try:
                normalized_action = self._normalize_action(raw_action, index, seen_action_ids)
                resolved_params = self._resolve_refs(normalized_action.params, outputs_by_action)
                tool = self.registry.get(normalized_action.tool)
                normalized_params = normalize_action_params(tool, resolved_params)
                repaired_messages = [repair.reason for repair in normalized_params.repairs]
                report_repairs.extend(repaired_messages)
                validated_params = tool.validate_params(normalized_params.params)
                self._maybe_inject_transient_failure(normalized_action.tool)

                log_action_start(
                    self.logger,
                    normalized_action.action_id,
                    normalized_action.tool,
                    validated_params,
                )
                tool_result = tool.execute(context, validated_params)
                outputs = dict(tool_result.outputs)
                results.append(
                    ActionExecutionResult(
                        action_id=normalized_action.action_id,
                        tool=normalized_action.tool,
                        success=True,
                        outputs=outputs,
                        repairs=repaired_messages,
                    )
                )
                if normalized_action.action_id is not None:
                    outputs_by_action[normalized_action.action_id] = outputs
                log_action_success(
                    self.logger,
                    normalized_action.action_id,
                    normalized_action.tool,
                    outputs,
                )
            except (ActionValidationError, ReferenceResolutionError, ToolRegistryError, ToolValidationError, ToolExecutionError) as exc:
                message = str(exc)
                log_action_failure(self.logger, action_id, tool_name, message)
                results.append(
                    ActionExecutionResult(
                        action_id=action_id,
                        tool=tool_name,
                        success=False,
                        error=message,
                        repairs=report_repairs,
                    )
                )
                report = ExecutionReport(
                    success=False,
                    results=results,
                    failed_action_id=action_id,
                    message=f"Action failed: {tool_name} - {message}",
                    repairs=report_repairs,
                )
                log_execution_report(self.logger, report)
                return report
            except Exception as exc:  # pragma: no cover - hard guard for Blender runtime
                message = f"Unexpected execution error: {exc}"
                log_action_failure(self.logger, action_id, tool_name, message)
                results.append(
                    ActionExecutionResult(
                        action_id=action_id,
                        tool=tool_name,
                        success=False,
                        error=message,
                        repairs=report_repairs,
                    )
                )
                report = ExecutionReport(
                    success=False,
                    results=results,
                    failed_action_id=action_id,
                    message=f"Action failed: {tool_name} - {message}",
                    repairs=report_repairs,
                )
                log_execution_report(self.logger, report)
                return report

        success_message = (
            "No actions to execute"
            if not results
            else f"Executed {len(results)} action(s) successfully"
        )
        report = ExecutionReport(success=True, results=results, message=success_message, repairs=report_repairs)
        log_execution_report(self.logger, report)
        return report

    def _maybe_inject_transient_failure(self, tool_name: str) -> None:
        configured = os.getenv("VECTRA_AUDIT_INJECT_TRANSIENT_FAILURE", "").strip()
        if not configured or configured != tool_name:
            return
        if tool_name in self._transient_failures_seen:
            return
        self._transient_failures_seen.add(tool_name)
        raise ToolExecutionError(f"Injected transient failure for audit on '{tool_name}'")

    @staticmethod
    def _extract_action_id(raw_action: Any) -> str | None:
        if isinstance(raw_action, Mapping):
            action_id = raw_action.get("action_id")
            if isinstance(action_id, str):
                return action_id
        return None

    @staticmethod
    def _extract_tool_name(raw_action: Any) -> str:
        if isinstance(raw_action, Mapping):
            tool_name = raw_action.get("tool")
            if isinstance(tool_name, str):
                return tool_name
        return "<invalid>"

    def _normalize_action(
        self,
        raw_action: Any,
        index: int,
        seen_action_ids: set[str],
    ) -> NormalizedAction:
        if not isinstance(raw_action, Mapping):
            raise ActionValidationError(f"Action at index {index} must be a mapping")

        tool_name = raw_action.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ActionValidationError(f"Action at index {index} must include a non-empty tool")

        params = raw_action.get("params", {})
        if not isinstance(params, Mapping):
            raise ActionValidationError(f"Action '{tool_name}' params must be a mapping")

        action_id = raw_action.get("action_id")
        if action_id is not None and (not isinstance(action_id, str) or not action_id.strip()):
            raise ActionValidationError(f"Action '{tool_name}' has an invalid action_id")

        if isinstance(action_id, str):
            if action_id in seen_action_ids:
                raise ActionValidationError(f"Duplicate action_id '{action_id}'")
            seen_action_ids.add(action_id)

        return NormalizedAction(
            action_id=action_id,
            tool=tool_name,
            params=dict(params),
        )

    def _resolve_refs(self, value: Any, outputs_by_action: dict[str, dict[str, Any]]) -> Any:
        if isinstance(value, Mapping):
            if set(value.keys()) == {"$ref"}:
                return self._resolve_single_ref(value["$ref"], outputs_by_action)
            return {
                key: self._resolve_refs(nested_value, outputs_by_action)
                for key, nested_value in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_refs(item, outputs_by_action) for item in value]
        return value

    def _resolve_single_ref(self, raw_ref: Any, outputs_by_action: dict[str, dict[str, Any]]) -> Any:
        if not isinstance(raw_ref, str) or "." not in raw_ref:
            raise ReferenceResolutionError("Reference values must use the format '<action_id>.<output_key>'")

        action_id, *path = raw_ref.split(".")
        if action_id not in outputs_by_action:
            raise ReferenceResolutionError(f"Unknown action reference '{action_id}'")

        current: Any = outputs_by_action[action_id]
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                raise ReferenceResolutionError(f"Reference '{raw_ref}' could not be resolved")
            current = current[part]
        return current
