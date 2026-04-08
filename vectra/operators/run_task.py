from __future__ import annotations

import os
import queue
import threading
import json
from dataclasses import dataclass, field
from typing import Any

import bpy

from ..agent.observation import build_scene_state as build_agent_scene_state
from ..agent.observation import capture_viewport_screenshot
from ..agent.reflection import summarize_scene_diff
from ..addon_bootstrap import current_dev_source_path
from ..bridge.client import BridgeClientError, create_agent_step, create_task
from ..execution import ConsoleCodeExecutor, ExecutionEngine
from ..runtime_service import ensure_local_backend
from ..utils.logging import get_vectra_logger, log_structured

logger = get_vectra_logger("vectra.blender")

DEFAULT_BASE_URL = os.getenv("VECTRA_BASE_URL", "http://127.0.0.1:8000").strip() or "http://127.0.0.1:8000"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_AGENT_ITERATIONS = 20
ALLOWED_PHASES = {"idle", "sending", "working", "clarifying", "success", "error"}
DEFAULT_EXECUTION_MODE = "vectra-dev"

_request_lock = threading.Lock()
_request_thread: threading.Thread | None = None
_request_queue: queue.Queue[Any] | None = None
_request_scene_name: str | None = None
_request_kind: str | None = None
_execution_engine: ExecutionEngine | None = None
_code_executor: ConsoleCodeExecutor | None = None


@dataclass
class AgentLoopState:
    prompt: str
    execution_mode: str
    history: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    failure_signatures: dict[str, int] = field(default_factory=dict)
    ineffective_turns: int = 0
    last_action_families: list[str] = field(default_factory=list)
    budget_state: dict[str, Any] = field(default_factory=dict)


_agent_loop_state: AgentLoopState | None = None


def _vector_to_list(vector: Any) -> list[float]:
    return [float(component) for component in vector[:3]]


def _build_scene_state(context: bpy.types.Context) -> dict[str, Any]:
    return build_agent_scene_state(context)


def _is_agent_mode_enabled() -> bool:
    raw_value = os.getenv("VECTRA_AGENT_MODE", "").strip().lower()
    if not raw_value:
        return True
    return raw_value in {"1", "true", "yes", "on"}


def _normalize_execution_mode(raw_value: Any) -> str:
    if isinstance(raw_value, str) and raw_value.strip() == "vectra-code":
        return "vectra-code"
    return DEFAULT_EXECUTION_MODE


def _set_phase(scene: bpy.types.Scene, phase: str) -> None:
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"Unsupported Vectra phase: {phase}")
    scene.vectra_phase = phase


def _set_runtime_state(scene: bpy.types.Scene, runtime_state: str) -> None:
    if hasattr(scene, "vectra_runtime_state"):
        scene.vectra_runtime_state = runtime_state


def _append_transcript(scene: bpy.types.Scene, message: str | None) -> None:
    if not message:
        return
    normalized = message.strip()
    if not normalized:
        return
    existing = getattr(scene, "vectra_agent_transcript", "")
    scene.vectra_agent_transcript = f"{existing}\n{normalized}".strip() if existing else normalized


def _sync_history(scene: bpy.types.Scene, state: AgentLoopState | None) -> None:
    history = state.history if state is not None else []
    scene.vectra_history_json = json.dumps(history)


def _apply_ui_state(
    scene: bpy.types.Scene,
    *,
    status: str,
    phase: str,
    runtime_state: str | None = None,
) -> None:
    _set_phase(scene, phase)
    scene.vectra_status = status
    if runtime_state is not None:
        _set_runtime_state(scene, runtime_state)


def _runtime_state_from_metadata(metadata: dict[str, Any] | None, default: str = "") -> str:
    if not isinstance(metadata, dict):
        return default
    runtime_state = metadata.get("runtime_state")
    if isinstance(runtime_state, str) and runtime_state.strip():
        return runtime_state.strip()
    return default


def _is_poll_timer_registered() -> bool:
    return bpy.app.timers.is_registered(_poll_request_result)


def _scene_from_name(scene_name: str | None) -> bpy.types.Scene | None:
    if scene_name:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            return scene
    return getattr(bpy.context, "scene", None)


def _get_execution_engine() -> ExecutionEngine:
    global _execution_engine

    if _execution_engine is None:
        _execution_engine = ExecutionEngine()
    return _execution_engine


def _get_code_executor() -> ConsoleCodeExecutor:
    global _code_executor

    if _code_executor is None:
        _code_executor = ConsoleCodeExecutor()
    return _code_executor


def _worker(
    payload: dict[str, Any],
    base_url: str,
    result_queue: queue.Queue[Any],
    *,
    request_kind: str,
) -> None:
    try:
        ensure_local_backend(
            base_url=base_url,
            repo_root_hint=current_dev_source_path(),
        )
        if request_kind == "agent_step":
            response = create_agent_step(
                payload,
                base_url=base_url,
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        else:
            response = create_task(
                payload,
                base_url=base_url,
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        result_queue.put((request_kind, "success", response))
    except BridgeClientError as exc:
        logger.warning("Vectra request failed: %s", exc)
        result_queue.put((request_kind, "error", exc))
    except Exception as exc:  # pragma: no cover - defensive safeguard for Blender runtime
        logger.exception("Unexpected Vectra request failure")
        result_queue.put((request_kind, "error", exc))


def _finalize_request() -> None:
    global _request_thread, _request_queue, _request_scene_name, _request_kind

    _request_thread = None
    _request_queue = None
    _request_scene_name = None
    _request_kind = None


def _history_entry(iteration: int, role: str, summary: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "role": role,
        "summary": summary,
        "details": details or {},
    }


def _execution_signature(execution_payload: dict[str, Any]) -> str:
    if execution_payload.get("kind") == "console_code":
        return str(execution_payload.get("code", "")).strip()
    return str(execution_payload.get("actions", []))


def _action_families(execution_payload: dict[str, Any]) -> list[str]:
    metadata = execution_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    families = metadata.get("action_families", [])
    if not isinstance(families, list):
        return []
    return [value for value in families if isinstance(value, str) and value.strip()]


def _start_worker(
    payload: dict[str, Any],
    *,
    request_kind: str,
    scene_name: str,
) -> None:
    global _request_thread, _request_queue, _request_scene_name, _request_kind

    result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=_worker,
        args=(payload, DEFAULT_BASE_URL, result_queue),
        kwargs={"request_kind": request_kind},
        daemon=True,
        name="vectra-request-thread",
    )

    _request_queue = result_queue
    _request_thread = worker
    _request_scene_name = scene_name
    _request_kind = request_kind
    worker.start()


def _start_legacy_request(context: bpy.types.Context) -> None:
    scene = context.scene
    payload = {
        "prompt": scene.vectra_prompt,
        "scene_state": _build_scene_state(context),
        "images": [],
    }
    scene.vectra_request_in_flight = True
    _apply_ui_state(
        scene,
        status="Awaiting model response...",
        phase="sending",
        runtime_state="awaiting_model_response",
    )
    _start_worker(payload, request_kind="legacy", scene_name=scene.name)


def _start_agent_iteration(context: bpy.types.Context) -> None:
    state = _agent_loop_state
    scene = context.scene
    if state is None:
        _apply_ui_state(scene, status="Agent loop state is missing", phase="error")
        scene.vectra_request_in_flight = False
        _finalize_request()
        return

    if state.iteration >= MAX_AGENT_ITERATIONS:
        _apply_ui_state(scene, status="Agent loop reached the maximum iteration limit", phase="error")
        scene.vectra_request_in_flight = False
        _finalize_request()
        return

    state.iteration += 1
    scene.vectra_iteration = state.iteration
    screenshot = capture_viewport_screenshot(state.iteration)
    payload = {
        "prompt": state.prompt,
        "scene_state": _build_scene_state(context),
        "screenshot": screenshot,
        "history": state.history,
        "iteration": state.iteration,
        "execution_mode": state.execution_mode,
    }
    scene.vectra_request_in_flight = True
    scene.vectra_pending_question = ""
    _apply_ui_state(
        scene,
        status="Awaiting model response...",
        phase="working",
        runtime_state="awaiting_model_response",
    )
    _start_worker(payload, request_kind="agent_step", scene_name=scene.name)


def _handle_legacy_result(scene: bpy.types.Scene, payload: dict[str, Any]) -> float | None:
    log_structured(logger, "backend_response", payload)
    actions = payload.get("actions", [])
    response_status = str(payload.get("status", "error")).strip().lower()
    planner_message = str(payload.get("message", "No actions returned")).strip()
    response_metadata = payload.get("metadata", {})
    runtime_state = _runtime_state_from_metadata(response_metadata, "valid_action_batch_ready" if actions else "")
    if response_status == "ok" and actions:
        try:
            _apply_ui_state(
                scene,
                status=str(response_metadata.get("runtime_state_detail", "Valid action batch ready")).strip()
                or "Valid action batch ready",
                phase="working",
                runtime_state=runtime_state or "valid_action_batch_ready",
            )
            report = _get_execution_engine().run(bpy.context, actions)
        except Exception:  # pragma: no cover - defensive safeguard for Blender runtime
            logger.exception("Unexpected Vectra execution failure")
            _apply_ui_state(
                scene,
                status="Execution failed",
                phase="error",
                runtime_state=runtime_state or "provider_transport_failure",
            )
        else:
            if report.success:
                _apply_ui_state(
                    scene,
                    status=f"Executed {len(report.results)} action(s) successfully",
                    phase="success",
                    runtime_state=runtime_state or "valid_action_batch_ready",
                )
            else:
                failure_status = report.message or f"Failed at {report.failed_action_id or 'unknown'}"
                _apply_ui_state(
                    scene,
                    status=failure_status,
                    phase="error",
                    runtime_state=runtime_state or "provider_transport_failure",
                )
    else:
        _apply_ui_state(
            scene,
            status=planner_message,
            phase="error",
            runtime_state=runtime_state or "provider_transport_failure",
        )

    scene.vectra_request_in_flight = False
    _finalize_request()
    return None


def _handle_agent_execution(scene: bpy.types.Scene, response: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    execution_payload = response.get("execution", {})
    if not isinstance(execution_payload, dict):
        return False, "Agent response was missing execution payload", {}

    kind = execution_payload.get("kind", "none")
    if kind == "tool_actions":
        actions = execution_payload.get("actions", [])
        report = _get_execution_engine().run(bpy.context, actions)
        affected_object_names = [
            result.outputs.get("object_name")
            for result in report.results
            if result.success and isinstance(result.outputs, dict) and isinstance(result.outputs.get("object_name"), str)
        ]
        return report.success, report.message, {
            "kind": kind,
            "results": [result.__dict__ for result in report.results],
            "actions": actions,
            "repairs": list(report.repairs),
            "metadata": {
                "affected_object_names": affected_object_names,
                **(execution_payload.get("metadata", {}) if isinstance(execution_payload.get("metadata"), dict) else {}),
            },
        }

    if kind == "console_code":
        code = str(execution_payload.get("code", "")).strip()
        report = _get_code_executor().run(bpy.context, code)
        return report.success, report.message, {
            "kind": kind,
            "code": code,
            "result": report.result,
        }

    return True, str(response.get("message", "No execution needed")).strip(), {
        "kind": "none",
        "actions": [],
    }


def _maybe_continue_agent_loop(
    scene: bpy.types.Scene,
    response: dict[str, Any],
    *,
    success: bool,
    execution_payload: dict[str, Any],
    verification: dict[str, Any],
) -> float | None:
    global _agent_loop_state

    state = _agent_loop_state
    if state is None:
        _apply_ui_state(scene, status="Agent loop state was lost", phase="error")
        scene.vectra_request_in_flight = False
        _finalize_request()
        return None

    signature = _execution_signature(execution_payload)
    action_families = _action_families(execution_payload)
    if not success:
        state.failure_signatures[signature] = state.failure_signatures.get(signature, 0) + 1
        if state.failure_signatures[signature] >= 2:
            _apply_ui_state(
                scene,
                status="Stopping after repeated invalid execution attempts",
                phase="error",
            )
            scene.vectra_request_in_flight = False
            _finalize_request()
            return None
    else:
        state.failure_signatures.pop(signature, None)
    progress_score = float(verification.get("progress_score", 0.0) or 0.0)
    effective_change = bool(verification.get("meaningful_change"))
    structural_progress = bool(verification.get("structural_progress"))
    low_progress = bool(verification.get("low_progress"))

    if not success:
        if state.iteration >= MAX_AGENT_ITERATIONS:
            _apply_ui_state(scene, status="Agent loop reached the maximum iteration limit", phase="error")
            scene.vectra_request_in_flight = False
            _finalize_request()
            return None
        _append_transcript(scene, "Adjusting after the last step did not land correctly.")
        state.ineffective_turns = min(state.ineffective_turns + 1, 999)
        state.last_action_families = action_families
        _start_agent_iteration(bpy.context)
        return 0.1

    if structural_progress:
        state.ineffective_turns = 0
        _append_transcript(
            scene,
            f"Partial structural progress counted with score {progress_score:.2f}; the next turn should build on it instead of restarting.",
        )
    elif not effective_change:
        state.ineffective_turns += 1
        repeated_family = bool(action_families) and action_families == state.last_action_families
        if repeated_family:
            _append_transcript(scene, "The last ineffective step reused the same action family, so the next turn must switch strategies.")
        else:
            _append_transcript(scene, "The last step did not produce a meaningful scene change, so the next turn must change tactics.")
        if state.ineffective_turns >= 3:
            _apply_ui_state(scene, status="Stopping after repeated ineffective turns", phase="error")
            scene.vectra_request_in_flight = False
            _finalize_request()
            return None
    elif low_progress:
        state.ineffective_turns = min(state.ineffective_turns + 1, 2)
        repeated_family = bool(action_families) and action_families == state.last_action_families
        if repeated_family:
            _append_transcript(
                scene,
                f"The last step made only limited progress (score {progress_score:.2f}) and reused the same action family, so the next turn must take a stronger scene-level swing.",
            )
        else:
            _append_transcript(
                scene,
                f"The last step made limited progress (score {progress_score:.2f}), so the next turn should use a stronger coordinated batch.",
            )
    else:
        state.ineffective_turns = 0

    state.last_action_families = action_families

    if response.get("status") == "complete" or not bool(response.get("continue_loop")):
        final_status = verification.get("summary", "") or str(response.get("message", "Task complete")).strip()
        _apply_ui_state(scene, status=final_status, phase="success")
        scene.vectra_request_in_flight = False
        _finalize_request()
        return None

    _apply_ui_state(scene, status="Reflecting and preparing the next step...", phase="working")
    _start_agent_iteration(bpy.context)
    return 0.1


def _handle_agent_result(scene: bpy.types.Scene, response: dict[str, Any]) -> float | None:
    global _agent_loop_state

    state = _agent_loop_state
    if state is None:
        _apply_ui_state(scene, status="Agent loop state is missing", phase="error")
        scene.vectra_request_in_flight = False
        _finalize_request()
        return None

    narration = str(response.get("narration", "")).strip()
    assumptions = response.get("assumptions", [])
    response_metadata = response.get("metadata", {})
    runtime_state = _runtime_state_from_metadata(response_metadata, "")
    runtime_state_detail = ""
    if isinstance(response_metadata, dict):
        runtime_state_detail = str(response_metadata.get("runtime_state_detail", "")).strip()
    if isinstance(response_metadata, dict):
        state.budget_state = dict(response_metadata.get("budget_state", {})) if isinstance(response_metadata.get("budget_state"), dict) else {}
    if narration:
        scene.vectra_status = narration
        _append_transcript(scene, narration)
    if isinstance(assumptions, list):
        assumption_lines = [
            f"Assumption: {item.get('reason', '')}"
            for item in assumptions
            if isinstance(item, dict) and str(item.get("reason", "")).strip()
        ]
        for line in assumption_lines[:3]:
            _append_transcript(scene, line)
    if isinstance(response_metadata, dict) and response_metadata.get("completion_mode_active"):
        _append_transcript(scene, "Completion mode is active, so the next steps should prioritize a coherent finish.")

    state.history.append(
        _history_entry(
            state.iteration,
            "agent",
            narration or str(response.get("message", "Planned next step")).strip(),
            {
                "understanding": response.get("understanding", ""),
                "plan": response.get("plan", []),
                "intended_actions": response.get("intended_actions", []),
                "assumptions": assumptions if isinstance(assumptions, list) else [],
                "preferred_execution_mode": response.get("preferred_execution_mode", DEFAULT_EXECUTION_MODE),
                "metadata": response_metadata if isinstance(response_metadata, dict) else {},
            },
        )
    )
    _sync_history(scene, state)

    response_status = str(response.get("status", "error")).strip().lower()
    if response_status == "clarify":
        question = str(response.get("question", "")).strip() or "I need clarification before I continue."
        scene.vectra_pending_question = question
        _append_transcript(scene, question)
        _apply_ui_state(
            scene,
            status=question,
            phase="clarifying",
            runtime_state=runtime_state or "valid_action_batch_ready",
        )
        scene.vectra_request_in_flight = False
        _finalize_request()
        return None

    if response_status == "error":
        message = str(response.get("message", "Agent step failed")).strip()
        _apply_ui_state(
            scene,
            status=message,
            phase="error",
            runtime_state=runtime_state or "provider_transport_failure",
        )
        scene.vectra_request_in_flight = False
        _finalize_request()
        return None

    if runtime_state:
        ready_status = runtime_state_detail or {
            "fallback_provider_invoked": "Valid action batch ready from fallback provider.",
            "valid_action_batch_ready": "Valid action batch ready.",
        }.get(runtime_state, narration or "Prepared the next step.")
        _apply_ui_state(
            scene,
            status=ready_status,
            phase="working",
            runtime_state=runtime_state,
        )
        if ready_status != narration:
            _append_transcript(scene, ready_status)

    before_scene = _build_scene_state(bpy.context)
    try:
        success, execution_message, execution_details = _handle_agent_execution(scene, response)
    except Exception as exc:  # pragma: no cover - defensive Blender runtime guard
        logger.exception("Unexpected Vectra agent execution failure")
        success = False
        execution_message = str(exc)
        execution_details = {"kind": "error", "message": execution_message}

    execution_payload = response.get("execution", {})
    normalized_execution_payload = execution_payload if isinstance(execution_payload, dict) else {}
    after_scene = _build_scene_state(bpy.context)
    verification = summarize_scene_diff(before_scene, after_scene, normalized_execution_payload)
    state.history.append(
        _history_entry(
            state.iteration,
            "execution",
            execution_message,
            execution_details,
        )
    )
    state.history.append(
        _history_entry(
            state.iteration,
            "verification",
            verification["summary"],
            verification,
        )
    )
    _sync_history(scene, state)
    _append_transcript(scene, execution_message)
    _append_transcript(scene, verification["summary"])

    execution_payload = response.get("execution", {})
    return _maybe_continue_agent_loop(
        scene,
        response,
        success=success,
        execution_payload=normalized_execution_payload,
        verification=verification,
    )


def _poll_request_result() -> float | None:
    result_queue = _request_queue
    if result_queue is None:
        _finalize_request()
        return None

    try:
        queued = result_queue.get_nowait()
    except queue.Empty:
        return 0.1

    if isinstance(queued, tuple) and len(queued) == 3:
        request_kind, result_type, payload = queued
    elif isinstance(queued, tuple) and len(queued) == 2:
        request_kind = _request_kind or "legacy"
        result_type, payload = queued
    else:  # pragma: no cover - defensive guard
        request_kind, result_type, payload = "legacy", "error", "Malformed queue payload"

    scene = _scene_from_name(_request_scene_name)
    if scene is None:
        _finalize_request()
        return None

    if result_type == "error":
        error_message = str(payload).strip() if payload is not None else "Connection failed"
        scene.vectra_request_in_flight = False
        _apply_ui_state(
            scene,
            status=error_message,
            phase="error",
            runtime_state="provider_transport_failure",
        )
        _finalize_request()
        return None

    if request_kind == "agent_step":
        return _handle_agent_result(scene, payload)
    return _handle_legacy_result(scene, payload)


def cleanup_request_state() -> None:
    global _execution_engine, _code_executor, _request_queue, _agent_loop_state

    scene = _scene_from_name(_request_scene_name)
    if scene is not None:
        if hasattr(scene, "vectra_request_in_flight"):
            scene.vectra_request_in_flight = False
        if hasattr(scene, "vectra_phase") and scene.vectra_phase in {"sending", "working", "clarifying"}:
            scene.vectra_phase = "idle"
            if hasattr(scene, "vectra_status"):
                scene.vectra_status = "Idle"
        if hasattr(scene, "vectra_runtime_state"):
            scene.vectra_runtime_state = "idle"
        if hasattr(scene, "vectra_pending_question"):
            scene.vectra_pending_question = ""
        if hasattr(scene, "vectra_iteration"):
            scene.vectra_iteration = 0
        if hasattr(scene, "vectra_history_json"):
            scene.vectra_history_json = "[]"

    _request_queue = None
    _execution_engine = None
    _code_executor = None
    _agent_loop_state = None
    if _is_poll_timer_registered():
        bpy.app.timers.unregister(_poll_request_result)
    _finalize_request()


def get_reload_block_reason() -> str | None:
    if _request_thread is not None and _request_thread.is_alive():
        return "Cannot reload Vectra while a request worker thread is still running"
    if _request_queue is not None:
        return "Cannot reload Vectra while request cleanup is still pending"
    if _is_poll_timer_registered():
        return "Cannot reload Vectra while the request poll timer is still registered"
    return None


class VECTRA_OT_run_task(bpy.types.Operator):
    bl_idname = "vectra.run_task"
    bl_label = "Run Task"
    bl_description = "Send the current prompt to the local Vectra runtime"

    def execute(self, context: bpy.types.Context) -> set[str]:
        global _agent_loop_state

        scene = context.scene
        if scene.vectra_request_in_flight:
            self.report({"INFO"}, "Vectra request already running")
            return {"CANCELLED"}

        with _request_lock:
            if _request_thread is not None and _request_thread.is_alive():
                scene.vectra_request_in_flight = True
                _set_phase(scene, "sending")
                scene.vectra_status = "Awaiting model response..."
                _set_runtime_state(scene, "awaiting_model_response")
                return {"CANCELLED"}

            scene.vectra_agent_transcript = ""
            scene.vectra_pending_question = ""
            scene.vectra_iteration = 0
            scene.vectra_history_json = "[]"
            _set_runtime_state(scene, "idle")
            if _is_poll_timer_registered():
                bpy.app.timers.unregister(_poll_request_result)
            bpy.app.timers.register(_poll_request_result, first_interval=0.1)

            if _is_agent_mode_enabled():
                _agent_loop_state = AgentLoopState(
                    prompt=scene.vectra_prompt,
                    execution_mode=_normalize_execution_mode(getattr(scene, "vectra_execution_mode", DEFAULT_EXECUTION_MODE)),
                )
                _start_agent_iteration(context)
            else:
                _start_legacy_request(context)

        return {"FINISHED"}
