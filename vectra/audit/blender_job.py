from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import addon_utils
import bpy

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vectra.agent.observation import build_scene_state, capture_viewport_screenshot


def _parse_config_path(argv: list[str]) -> Path:
    if "--" not in argv:
        raise RuntimeError("Expected '-- <config-path>' after Blender arguments")
    marker = argv.index("--")
    if marker + 1 >= len(argv):
        raise RuntimeError("Expected a config path after '--'")
    return Path(argv[marker + 1]).expanduser().resolve()


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _add_named_primitive(primitive_type: str, *, name: str, location: tuple[float, float, float], scale: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    if primitive_type == "cube":
        bpy.ops.mesh.primitive_cube_add(location=location)
    elif primitive_type == "plane":
        bpy.ops.mesh.primitive_plane_add(location=location)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(location=location)
    obj = bpy.context.active_object
    if obj is not None:
        obj.name = name
        obj.scale = scale


def _ensure_light(name: str, location: tuple[float, float, float], energy: float) -> None:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.active_object
    if light is not None:
        light.name = name
        light.data.energy = energy


def _ensure_camera(name: str, location: tuple[float, float, float], rotation: tuple[float, float, float]) -> None:
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object
    if camera is not None:
        camera.name = name
        bpy.context.scene.camera = camera


def _seed_scene(setup_id: str) -> None:
    _clear_scene()
    if setup_id == "empty-scene":
        return
    if setup_id == "plain-geometric-scene":
        _add_named_primitive("plane", name="Floor", location=(0.0, 0.0, -1.0), scale=(6.0, 6.0, 1.0))
        _add_named_primitive("cube", name="CenterCube", location=(0.0, 0.0, 0.0))
        _add_named_primitive("sphere", name="AccentSphere", location=(2.5, 0.0, 1.0), scale=(0.6, 0.6, 0.6))
        return
    if setup_id == "cluttered-irrelevant-object-scene":
        _add_named_primitive("plane", name="MessyFloor", location=(0.0, 0.0, -1.0), scale=(8.0, 8.0, 1.0))
        for index, location in enumerate(((-2.0, 0.0, 0.0), (-1.0, -1.0, 0.0), (0.5, 1.5, 0.0), (1.0, -0.5, 0.0), (2.0, 0.5, 0.0)), start=1):
            _add_named_primitive("cube", name=f"Clutter_{index}", location=location, scale=(0.7, 0.7, 0.7))
        return
    if setup_id == "half-built-scene":
        _add_named_primitive("plane", name="BaseFloor", location=(0.0, 0.0, -1.0), scale=(8.0, 8.0, 1.0))
        _add_named_primitive("cube", name="ProtoPart", location=(0.0, 0.0, 0.0), scale=(1.2, 1.2, 0.3))
        return
    if setup_id == "intentionally-awkward-layout-scene":
        _add_named_primitive("plane", name="AwkwardFloor", location=(0.0, 0.0, -1.0), scale=(6.0, 6.0, 1.0))
        for index, location in enumerate(((0.0, 0.0, 0.0), (0.1, 0.2, 0.0), (0.2, -0.1, 0.0)), start=1):
            _add_named_primitive("cube", name=f"Overlap_{index}", location=location)
        return
    if setup_id == "simple-lit-scene-for-animation":
        _add_named_primitive("plane", name="Stage", location=(0.0, 0.0, -1.0), scale=(6.0, 6.0, 1.0))
        _add_named_primitive("cube", name="Hero", location=(0.0, 0.0, 0.0))
        _ensure_light("KeyLight", (4.0, -4.0, 6.0), 1500.0)
        _ensure_camera("MainCamera", (8.0, -8.0, 6.0), (1.1, 0.0, 0.8))
        return
    if setup_id == "transient-execution-failure-scene":
        _add_named_primitive("plane", name="TransientFloor", location=(0.0, 0.0, -1.0), scale=(6.0, 6.0, 1.0))
        _add_named_primitive("cube", name="TransientCube", location=(0.0, 0.0, 0.0))
        return
    raise RuntimeError(f"Unknown audit setup '{setup_id}'")


def _quality_flags(after_state: dict[str, Any]) -> dict[str, Any]:
    objects = after_state.get("objects", [])
    lights = after_state.get("lights", [])
    groups = after_state.get("groups", [])
    return {
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "has_light": bool(lights),
        "has_camera": bool(after_state.get("active_camera")),
        "has_groups": bool(groups),
        "has_animation": any(bool(obj.get("has_animation")) for obj in objects if isinstance(obj, dict)),
    }


def _collect_step_screenshots() -> list[str]:
    screenshot_dir = Path(tempfile.gettempdir()) / "vectra-agent-screenshots"
    if not screenshot_dir.is_dir():
        return []
    return sorted(str(path) for path in screenshot_dir.glob("vectra-loop-*.png"))


def _build_artifact(config: dict[str, Any], *, started_at: float, before_state: dict[str, Any], final_screenshot: dict[str, Any]) -> dict[str, Any]:
    scene = bpy.context.scene
    after_state = build_scene_state(bpy.context)
    try:
        history_entries = json.loads(getattr(scene, "vectra_history_json", "[]"))
    except json.JSONDecodeError:
        history_entries = []
    verification_entries = [entry for entry in history_entries if isinstance(entry, dict) and entry.get("role") == "verification"]
    execution_entries = [entry for entry in history_entries if isinstance(entry, dict) and entry.get("role") == "execution"]
    agent_entries = [entry for entry in history_entries if isinstance(entry, dict) and entry.get("role") == "agent"]
    action_counts = [
        len(entry.get("details", {}).get("actions", []))
        for entry in execution_entries
        if isinstance(entry.get("details", {}), dict)
    ]
    wasted_turns = sum(
        1
        for entry in verification_entries
        if isinstance(entry.get("details", {}), dict) and not bool(entry["details"].get("meaningful_change"))
    )
    bulk_actions = 0
    micro_actions = 0
    provider_chain: list[str] = []
    for entry in agent_entries:
        details = entry.get("details", {})
        metadata = details.get("metadata", {}) if isinstance(details, dict) else {}
        if isinstance(metadata, dict):
            chains = metadata.get("provider_chain", [])
            if isinstance(chains, list):
                for value in chains:
                    if isinstance(value, str) and value not in provider_chain:
                        provider_chain.append(value)
    for entry in execution_entries:
        details = entry.get("details", {})
        actions = details.get("actions", []) if isinstance(details, dict) else []
        if not isinstance(actions, list):
            continue
        for action in actions:
            tool_name = action.get("tool", "") if isinstance(action, dict) else ""
            if any(tool_name.startswith(prefix) for prefix in ("object.transform_many", "object.delete_many", "object.distribute", "object.align", "scene.group", "object.parent")):
                bulk_actions += 1
            elif isinstance(tool_name, str) and tool_name:
                micro_actions += 1

    return {
        "prompt": config["prompt"],
        "mode": config["mode"],
        "setup_id": config["setup_id"],
        "provider_chain": provider_chain,
        "turns_used": int(getattr(scene, "vectra_iteration", 0)),
        "actions_per_turn": action_counts,
        "wall_time_seconds": round(time.time() - started_at, 3),
        "screenshots": _collect_step_screenshots() + ([final_screenshot["path"]] if final_screenshot.get("available") and final_screenshot.get("path") else []),
        "before_scene_state": before_state,
        "after_scene_state": after_state,
        "per_turn_diffs": verification_entries,
        "completion_status": {"phase": scene.vectra_phase, "status": scene.vectra_status},
        "wasted_turns": wasted_turns,
        "repeated_ineffective_action_count": max(0, wasted_turns - 1),
        "batch_size_statistics": {
            "min": min(action_counts) if action_counts else 0,
            "max": max(action_counts) if action_counts else 0,
            "average": (sum(action_counts) / float(len(action_counts))) if action_counts else 0.0,
        },
        "bulk_vs_micro_action_ratio": {
            "bulk_actions": bulk_actions,
            "micro_actions": micro_actions,
        },
        "quality_flags": _quality_flags(after_state),
        "transcript": scene.vectra_agent_transcript,
        "pending_question": scene.vectra_pending_question,
        "timed_out": bool(config.get("timed_out", False)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _finish(config: dict[str, Any], *, started_at: float, before_state: dict[str, Any]) -> None:
    final_screenshot = capture_viewport_screenshot(int(getattr(bpy.context.scene, "vectra_iteration", 0)) + 1)
    artifact = _build_artifact(config, started_at=started_at, before_state=before_state, final_screenshot=final_screenshot)
    _write_json(Path(config["artifact_path"]), artifact)
    bpy.ops.wm.quit_blender()


def main() -> None:
    config_path = _parse_config_path(sys.argv)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    os.environ["VECTRA_BASE_URL"] = str(config["base_url"])
    os.environ["VECTRA_AGENT_MODE"] = "1"
    if config["setup_id"] == "transient-execution-failure-scene":
        os.environ["VECTRA_AUDIT_INJECT_TRANSIENT_FAILURE"] = "object.transform"
    else:
        os.environ.pop("VECTRA_AUDIT_INJECT_TRANSIENT_FAILURE", None)

    addon_utils.enable("vectra", default_set=False, persistent=False)
    scene = bpy.context.scene
    _seed_scene(config["setup_id"])
    before_state = build_scene_state(bpy.context)
    scene.vectra_prompt = str(config["prompt"])
    scene.vectra_execution_mode = str(config["mode"])
    started_at = time.time()
    bpy.ops.vectra.run_task()

    def _poll() -> float | None:
        current_scene = bpy.context.scene
        if time.time() - started_at > float(config.get("timeout_seconds", 110.0)):
            config["timed_out"] = True
            current_scene.vectra_status = "Audit timed out"
            current_scene.vectra_phase = "error"
            _finish(config, started_at=started_at, before_state=before_state)
            return None
        if not current_scene.vectra_request_in_flight and current_scene.vectra_phase in {"success", "error", "clarifying"}:
            _finish(config, started_at=started_at, before_state=before_state)
            return None
        return 0.2

    bpy.app.timers.register(_poll, first_interval=0.2)


if __name__ == "__main__":
    main()
