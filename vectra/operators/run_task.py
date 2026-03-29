from __future__ import annotations

import queue
import threading
from typing import Any

import bpy

from ..bridge.client import BridgeClientError, create_task
from ..execution.engine import ExecutionEngine
from ..utils.logging import get_vectra_logger

logger = get_vectra_logger("vectra.blender")

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ALLOWED_PHASES = {"idle", "sending", "success", "error"}

_request_lock = threading.Lock()
_request_thread: threading.Thread | None = None
_request_queue: queue.Queue[tuple[str, Any]] | None = None
_request_scene_name: str | None = None
_execution_engine: ExecutionEngine | None = None


def _set_phase(scene: bpy.types.Scene, phase: str) -> None:
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"Unsupported Vectra phase: {phase}")
    scene.vectra_phase = phase


def _apply_ui_state(scene: bpy.types.Scene, *, status: str, phase: str) -> None:
    print("VECTRA DEBUG: setting status =", status)
    print("VECTRA DEBUG: setting phase =", phase)
    _set_phase(scene, phase)
    scene.vectra_status = status


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


def _worker(payload: dict[str, Any], base_url: str, result_queue: queue.Queue[tuple[str, Any]]) -> None:
    try:
        response = create_task(payload, base_url=base_url, timeout=5.0)
        result_queue.put(("success", response))
    except BridgeClientError as exc:
        logger.warning("Vectra request failed: %s", exc)
        result_queue.put(("error", exc))
    except Exception as exc:  # pragma: no cover - defensive safeguard for Blender runtime
        logger.exception("Unexpected Vectra request failure")
        result_queue.put(("error", exc))


def _finalize_request() -> None:
    global _request_thread, _request_queue, _request_scene_name

    _request_thread = None
    _request_queue = None
    _request_scene_name = None


def _poll_request_result() -> float | None:
    global _request_queue

    result_queue = _request_queue
    if result_queue is None:
        _finalize_request()
        return None

    try:
        result_type, payload = result_queue.get_nowait()
    except queue.Empty:
        return 0.1

    scene = _scene_from_name(_request_scene_name)
    if scene is not None:
        scene.vectra_request_in_flight = False
        if result_type == "success":
            response = payload
            print("VECTRA DEBUG: backend response =", response)
            actions = payload.get("actions", [])
            print("VECTRA DEBUG: actions =", actions)
            print("VECTRA DEBUG: actions type =", type(actions).__name__)
            if actions:
                try:
                    print("VECTRA DEBUG: about to run execution engine")
                    report = _get_execution_engine().run(bpy.context, actions)
                    print("VECTRA DEBUG: execution report =", report)
                except Exception:  # pragma: no cover - defensive safeguard for Blender runtime
                    logger.exception("Unexpected Vectra execution failure")
                    _apply_ui_state(scene, status="Execution failed", phase="error")
                else:
                    if report.success:
                        _apply_ui_state(
                            scene,
                            status=f"Executed {len(report.results)} action(s) successfully",
                            phase="success",
                        )
                    else:
                        _apply_ui_state(
                            scene,
                            status=f"Failed at {report.failed_action_id or 'unknown'}",
                            phase="error",
                        )
            else:
                _apply_ui_state(
                    scene,
                    status=f"No actions returned: {payload.get('message', 'No actions returned')}",
                    phase="success",
                )
        else:
            _apply_ui_state(scene, status="Connection failed", phase="error")

    _finalize_request()
    return None


def cleanup_request_state() -> None:
    global _execution_engine, _request_queue

    _request_queue = None
    _execution_engine = None
    if _is_poll_timer_registered():
        bpy.app.timers.unregister(_poll_request_result)
    _finalize_request()


class VECTRA_OT_run_task(bpy.types.Operator):
    bl_idname = "vectra.run_task"
    bl_label = "Run Task"
    bl_description = "Send the current prompt to the local Vectra runtime"

    def execute(self, context: bpy.types.Context) -> set[str]:
        global _request_thread, _request_queue, _request_scene_name

        scene = context.scene
        if scene.vectra_request_in_flight:
            self.report({"INFO"}, "Vectra request already running")
            return {"CANCELLED"}

        with _request_lock:
            if _request_thread is not None and _request_thread.is_alive():
                scene.vectra_request_in_flight = True
                _set_phase(scene, "sending")
                scene.vectra_status = "Sending request..."
                return {"CANCELLED"}

            payload = {
                "prompt": scene.vectra_prompt,
                "scene_state": {
                    "active_object": context.active_object.name if context.active_object else None,
                    "selected_objects": [obj.name for obj in context.selected_objects],
                    "current_frame": scene.frame_current,
                },
                "images": [],
            }

            result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
            worker = threading.Thread(
                target=_worker,
                args=(payload, DEFAULT_BASE_URL, result_queue),
                daemon=True,
                name="vectra-request-thread",
            )

            _request_queue = result_queue
            _request_thread = worker
            _request_scene_name = scene.name

            scene.vectra_request_in_flight = True
            _set_phase(scene, "sending")
            scene.vectra_status = "Sending request..."

            if _is_poll_timer_registered():
                bpy.app.timers.unregister(_poll_request_result)
            bpy.app.timers.register(_poll_request_result, first_interval=0.1)

            worker.start()

        return {"FINISHED"}
