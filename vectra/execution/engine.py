from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from typing import Any

from .normalization import normalize_action_params
from ..tools.base import ToolExecutionError, ToolValidationError
from ..tools.registry import ToolNotFoundError, ToolRegistry, ToolRegistryError, get_default_registry
from ..tools.spatial_constraints import SpatialConstraintSolver
from ..utils.logging import (
    get_vectra_logger,
    log_action_failure,
    log_action_start,
    log_action_success,
    log_execution_report,
    log_structured,
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
    status: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    message: str = ""
    missing_params: list[str] = field(default_factory=list)
    invalid_params: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "success" if self.success else "error"
        if self.success and not self.result:
            self.result = dict(self.outputs)
        if not self.success and self.error and not self.message:
            self.message = self.error


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

    def run(self, context: Any, actions: Any) -> ExecutionReport:
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes, bytearray)):
            result = self._error_result(
                action_id=None,
                tool="<invalid>",
                error_type="action_validation_failure",
                message="Actions must be a sequence",
                details={"received_type": type(actions).__name__},
            )
            report = ExecutionReport(
                success=False,
                results=[result],
                message="Action failed: <invalid> - Actions must be a sequence",
            )
            log_execution_report(self.logger, report)
            return report

        preflight_result = self._preflight_actions(context, actions)
        if preflight_result is not None:
            report = ExecutionReport(
                success=False,
                results=[preflight_result],
                failed_action_id=preflight_result.action_id,
                message=f"Action failed: {preflight_result.tool} - {preflight_result.message or preflight_result.error or 'Unknown error'}",
                repairs=list(preflight_result.repairs),
            )
            log_execution_report(self.logger, report)
            return report

        results: list[ActionExecutionResult] = []
        outputs_by_action: dict[str, dict[str, Any]] = {}
        seen_action_ids: set[str] = set()
        report_repairs: list[str] = []

        for index, raw_action in enumerate(actions):
            result = self.execute_action(
                context,
                raw_action,
                index=index,
                outputs_by_action=outputs_by_action,
                seen_action_ids=seen_action_ids,
            )
            results.append(result)
            report_repairs.extend(result.repairs)
            if not result.success:
                report = ExecutionReport(
                    success=False,
                    results=results,
                    failed_action_id=result.action_id,
                    message=f"Action failed: {result.tool} - {result.message or result.error or 'Unknown error'}",
                    repairs=report_repairs,
                )
                log_execution_report(self.logger, report)
                return report
            if result.action_id is not None:
                outputs_by_action[result.action_id] = dict(result.outputs)

        success_message = (
            "No actions to execute"
            if not results
            else f"Executed {len(results)} action(s) successfully"
        )
        report = ExecutionReport(success=True, results=results, message=success_message, repairs=report_repairs)
        log_execution_report(self.logger, report)
        return report

    def execute_action(
        self,
        context: Any,
        action: Any,
        *,
        index: int = 0,
        outputs_by_action: dict[str, dict[str, Any]] | None = None,
        seen_action_ids: set[str] | None = None,
    ) -> ActionExecutionResult:
        outputs = outputs_by_action if outputs_by_action is not None else {}
        seen_ids = seen_action_ids if seen_action_ids is not None else set()
        action_id = self._extract_action_id(action)
        tool_name = self._extract_tool_name(action)
        repaired_messages: list[str] = []

        try:
            normalized_action = self._normalize_action(action, index, seen_ids)
            tool_name = normalized_action.tool
            action_id = normalized_action.action_id
            resolved_params = self._resolve_refs(normalized_action.params, outputs)
            tool = self.registry.get(normalized_action.tool)
            repaired_params, auto_repairs = self._apply_auto_repairs(
                context,
                normalized_action.tool,
                resolved_params,
                normalized_action.action_id,
            )
            repaired_messages.extend(auto_repairs)
            normalized_params = normalize_action_params(tool, repaired_params)
            repaired_messages.extend(repair.reason for repair in normalized_params.repairs)
            validated_params = tool.validate_params(normalized_params.params)
            self._maybe_inject_transient_failure(normalized_action.tool)

            log_action_start(
                self.logger,
                normalized_action.action_id,
                normalized_action.tool,
                validated_params,
            )
            tool_result = tool.execute(context, validated_params)
            action_outputs = dict(tool_result.outputs)
            spatial_repairs = self._enforce_spatial_constraints(
                context,
                normalized_action.tool,
                action_outputs,
                normalized_action.action_id,
            )
            repaired_messages.extend(spatial_repairs)
            log_action_success(
                self.logger,
                normalized_action.action_id,
                normalized_action.tool,
                action_outputs,
            )
            if normalized_action.action_id is not None and outputs_by_action is not None:
                outputs_by_action[normalized_action.action_id] = action_outputs
            return ActionExecutionResult(
                action_id=normalized_action.action_id,
                tool=normalized_action.tool,
                success=True,
                outputs=action_outputs,
                repairs=repaired_messages,
                status="success",
                result=action_outputs,
                message=tool_result.message,
            )
        except ActionValidationError as exc:
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="action_validation_failure",
                message=str(exc),
                repairs=repaired_messages,
            )
        except ReferenceResolutionError as exc:
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="reference_resolution_failure",
                message=str(exc),
                repairs=repaired_messages,
            )
        except ToolNotFoundError as exc:
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="unknown_tool",
                message=str(exc),
                repairs=repaired_messages,
                details={"available_tools": self.registry.list_tools()},
            )
        except ToolValidationError as exc:
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="tool_validation_failure",
                message=str(exc),
                repairs=repaired_messages,
                missing_params=getattr(exc, "missing_params", []),
                invalid_params=getattr(exc, "invalid_params", []),
                details=getattr(exc, "details", {}),
            )
        except ToolExecutionError as exc:
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="tool_execution_failure",
                message=str(exc),
                repairs=repaired_messages,
            )
        except ToolRegistryError as exc:
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="unknown_tool",
                message=str(exc),
                repairs=repaired_messages,
            )
        except Exception as exc:  # pragma: no cover - hard guard for Blender runtime
            return self._error_result(
                action_id=action_id,
                tool=tool_name,
                error_type="unexpected_execution_error",
                message=f"Unexpected execution error: {exc}",
                repairs=repaired_messages,
                details={"exception_type": type(exc).__name__},
            )

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

    def _error_result(
        self,
        *,
        action_id: str | None,
        tool: str,
        error_type: str,
        message: str,
        repairs: list[str] | None = None,
        missing_params: list[str] | None = None,
        invalid_params: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> ActionExecutionResult:
        structured_details = dict(details or {})
        result = ActionExecutionResult(
            action_id=action_id,
            tool=tool,
            success=False,
            error=message,
            repairs=list(repairs or []),
            status="error",
            error_type=error_type,
            message=message,
            missing_params=list(missing_params or []),
            invalid_params=list(invalid_params or []),
            details=structured_details,
        )
        log_action_failure(self.logger, action_id, tool, message)
        log_event = {
            "tool_validation_failure": "tool_validation_failure",
            "unknown_tool": "unknown_tool_call",
            "tool_execution_failure": "tool_execution_error",
            "unexpected_execution_error": "tool_execution_error",
        }.get(error_type)
        if log_event is not None:
            log_structured(
                self.logger,
                log_event,
                {
                    "action_id": action_id,
                    "tool": tool,
                    "error_type": error_type,
                    "message": message,
                    "missing_params": result.missing_params,
                    "invalid_params": result.invalid_params,
                    "details": structured_details,
                },
                level="warning" if error_type in {"tool_validation_failure", "unknown_tool"} else "error",
            )
        return result

    def _preflight_actions(self, context: Any, actions: Sequence[Any]) -> ActionExecutionResult | None:
        seen_action_ids: set[str] = set()
        for index, raw_action in enumerate(actions):
            action_id = self._extract_action_id(raw_action)
            tool_name = self._extract_tool_name(raw_action)
            try:
                normalized_action = self._normalize_action(raw_action, index, seen_action_ids)
                tool_name = normalized_action.tool
                action_id = normalized_action.action_id
                tool = self.registry.get(normalized_action.tool)
                repaired_params, _auto_repairs = self._apply_auto_repairs(
                    context,
                    normalized_action.tool,
                    normalized_action.params,
                    normalized_action.action_id,
                    log_repair=False,
                )
                normalized_params = normalize_action_params(tool, repaired_params)
                validation_params = self._params_with_ref_placeholders(tool, normalized_params.params)
                tool.validate_params(validation_params)
            except ActionValidationError as exc:
                return self._error_result(
                    action_id=action_id,
                    tool=tool_name,
                    error_type="action_validation_failure",
                    message=str(exc),
                )
            except ToolNotFoundError as exc:
                return self._error_result(
                    action_id=action_id,
                    tool=tool_name,
                    error_type="unknown_tool",
                    message=str(exc),
                    details={"available_tools": self.registry.list_tools()},
                )
            except ToolValidationError as exc:
                return self._error_result(
                    action_id=action_id,
                    tool=tool_name,
                    error_type="tool_validation_failure",
                    message=str(exc),
                    missing_params=getattr(exc, "missing_params", []),
                    invalid_params=getattr(exc, "invalid_params", []),
                    details=getattr(exc, "details", {}),
                )
            except Exception as exc:  # pragma: no cover - preflight hard guard
                return self._error_result(
                    action_id=action_id,
                    tool=tool_name,
                    error_type="unexpected_execution_error",
                    message=f"Unexpected execution preflight error: {exc}",
                    details={"exception_type": type(exc).__name__},
                )
        return None

    def _enforce_spatial_constraints(
        self,
        context: Any,
        tool_name: str,
        action_outputs: dict[str, Any],
        action_id: str | None,
    ) -> list[str]:
        if not self._spatial_gate_applies(tool_name):
            return []
        objects = self._scene_objects(context)
        if not objects:
            return []
        affected_names = [
            name
            for name in action_outputs.get("object_names", [])
            if isinstance(name, str) and name.strip()
        ]
        object_name = action_outputs.get("object_name")
        if isinstance(object_name, str) and object_name.strip() and object_name not in affected_names:
            affected_names.append(object_name)
        if not affected_names:
            return []

        solver = SpatialConstraintSolver(objects)
        report = solver.validate(affected_names=affected_names)
        if report.ok:
            return []

        repairs: list[str] = []
        for repair_action in report.repair_actions:
            repair_tool_name = repair_action.get("tool")
            repair_params = repair_action.get("params", {})
            if not isinstance(repair_tool_name, str) or not isinstance(repair_params, dict):
                continue
            try:
                repair_tool = self.registry.get(repair_tool_name)
                normalized_params = normalize_action_params(repair_tool, repair_params)
                validated_params = repair_tool.validate_params(normalized_params.params)
                repair_result = repair_tool.execute(context, validated_params)
            except Exception as exc:
                repairs.append(f"Spatial repair {repair_tool_name} failed: {exc}")
                continue
            repaired_names = repair_result.outputs.get("object_names", [])
            repairs.append(
                f"Applied spatial repair {repair_tool_name} to {repaired_names or repair_params.get('target')}"
            )

        repaired_report = SpatialConstraintSolver(self._scene_objects(context)).validate(affected_names=affected_names)
        if repaired_report.ok:
            for message in repairs:
                log_structured(
                    self.logger,
                    "tool_auto_repair",
                    {
                        "action_id": action_id,
                        "tool": tool_name,
                        "repair": message,
                        "repair_type": "spatial_constraint",
                    },
                )
            return repairs

        issue_text = "; ".join(
            f"{issue.get('object')}: {issue.get('issues')}"
            for issue in repaired_report.top_issues
        )
        raise ToolExecutionError(f"Spatial constraints failed after repair: {issue_text}")

    @staticmethod
    def _spatial_gate_applies(tool_name: str) -> bool:
        return (
            tool_name == "mesh.create_primitive"
            or tool_name == "object.transform"
            or tool_name.startswith("object.place_")
            or tool_name in {"object.snap_to_support", "object.resolve_overlap", "object.fit_inside"}
        )

    @staticmethod
    def _scene_objects(context: Any) -> list[Any]:
        scene = getattr(context, "scene", None)
        objects = getattr(scene, "objects", None)
        if objects is None:
            return []
        try:
            return [
                obj for obj in list(objects)
                if hasattr(obj, "location") and hasattr(obj, "dimensions")
            ]
        except TypeError:
            return []

    def _apply_auto_repairs(
        self,
        context: Any,
        tool_name: str,
        params: Mapping[str, Any],
        action_id: str | None,
        *,
        log_repair: bool = True,
    ) -> tuple[dict[str, Any], list[str]]:
        repaired = dict(params)
        repairs: list[str] = []
        if tool_name != "object.place_on_surface":
            return repaired, repairs
        if "reference" in repaired and repaired["reference"] is not None:
            return repaired, repairs

        floor_name = self._exact_floor_name(context)
        if floor_name is None:
            return repaired, repairs

        repaired["reference"] = floor_name
        reason = f"Auto-filled missing reference with exact floor object '{floor_name}'"
        repairs.append(reason)
        if log_repair:
            log_structured(
                self.logger,
                "tool_auto_repair",
                {
                    "action_id": action_id,
                    "tool": tool_name,
                    "field": "reference",
                    "value": floor_name,
                    "reason": reason,
                },
            )
        return repaired, repairs

    def _params_with_ref_placeholders(self, tool: Any, params: dict[str, Any]) -> dict[str, Any]:
        validation_params = dict(params)
        for key, value in params.items():
            if self._contains_action_ref(value):
                spec = tool.input_schema.get(key, {})
                validation_params[key] = self._placeholder_for_spec(spec if isinstance(spec, dict) else {})
        return validation_params

    @staticmethod
    def _contains_action_ref(value: Any) -> bool:
        if isinstance(value, Mapping):
            if set(value.keys()) == {"$ref"}:
                return True
            return any(ExecutionEngine._contains_action_ref(nested_value) for nested_value in value.values())
        if isinstance(value, list):
            return any(ExecutionEngine._contains_action_ref(item) for item in value)
        return False

    @staticmethod
    def _placeholder_for_spec(spec: dict[str, Any]) -> Any:
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return enum_values[0]
        expected_type = str(spec.get("type", "string")).strip()
        if expected_type == "number":
            return 0.0
        if expected_type == "integer":
            return 0
        if expected_type == "boolean":
            return False
        if expected_type == "vector3":
            return [0.0, 0.0, 0.0]
        if expected_type == "string_array":
            return ["__pending_action_ref__"]
        return "__pending_action_ref__"

    @staticmethod
    def _exact_floor_name(context: Any) -> str | None:
        scene = getattr(context, "scene", None)
        objects = getattr(scene, "objects", None)
        if objects is None:
            return None
        floor_names = [
            name
            for obj in list(objects)
            if isinstance((name := getattr(obj, "name", None)), str) and name.lower() == "floor"
        ]
        if len(floor_names) == 1:
            return floor_names[0]
        return None

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


def execute_action(
    action: Any,
    *,
    context: Any,
    registry: ToolRegistry | None = None,
    logger: Any = None,
) -> ActionExecutionResult:
    return ExecutionEngine(registry=registry, logger=logger).execute_action(context, action)
