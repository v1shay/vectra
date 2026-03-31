from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent_runtime.planner as planner_module
from vectra.bridge.client import BridgeClientError, create_task, health_check

DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_BLENDER_BIN = "/Applications/Blender.app/Contents/MacOS/Blender"


class VerificationError(RuntimeError):
    """Raised when live verification fails."""


def _check_python_environment(python_bin: Path) -> None:
    if not python_bin.exists():
        raise VerificationError(f"Python executable not found: {python_bin}")

    completed = subprocess.run(
        [
            str(python_bin),
            "-c",
            (
                "import sys, fastapi, uvicorn, httpx, pydantic; "
                "print(sys.executable); "
                "print(fastapi.__version__); "
                "print(uvicorn.__version__); "
                "print(httpx.__version__); "
                "print(pydantic.__version__)"
            ),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"Python environment check failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _build_addon_archive(python_bin: Path) -> Path:
    completed = subprocess.run(
        [str(python_bin), str(REPO_ROOT / "scripts" / "package_addon.py")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"Failed to build add-on archive:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    zip_path = REPO_ROOT / "vectra_addon.zip"
    if not zip_path.exists():
        raise VerificationError(f"Expected add-on archive was not created: {zip_path}")
    return zip_path


def _read_process_output(process: subprocess.Popen[str], sink: list[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        sink.append(line)


def _start_backend(python_bin: Path, host: str, port: int) -> tuple[subprocess.Popen[str], list[str]]:
    backend_env = os.environ.copy()
    backend_env["PYTHONUNBUFFERED"] = "1"
    backend = subprocess.Popen(
        [
            str(python_bin),
            "-m",
            "uvicorn",
            "main:app",
            "--reload",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT / "agent_runtime",
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    backend_output: list[str] = []
    thread = threading.Thread(
        target=_read_process_output,
        args=(backend, backend_output),
        daemon=True,
        name="vectra-backend-log-reader",
    )
    thread.start()

    base_url = f"http://{host}:{port}"
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if backend.poll() is not None:
            raise VerificationError(
                f"Backend exited early while starting.\n{''.join(backend_output)}"
            )
        try:
            if health_check(base_url=base_url, timeout=0.5) == {"status": "ok"}:
                return backend, backend_output
        except Exception:
            time.sleep(0.1)

    raise VerificationError(
        f"Timed out waiting for backend health check.\n{''.join(backend_output)}"
    )


def _stop_backend(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def _assert_live_backend_round_trip(base_url: str) -> None:
    probe_response = create_task(
        {
            "prompt": "create a cube",
            "scene_state": {
                "active_object": None,
                "selected_objects": [],
                "current_frame": 1,
                "objects": [],
            },
            "images": [],
        },
        base_url=base_url,
        timeout=20.0,
    )
    if probe_response.get("status") not in {"ok", "error"}:
        raise VerificationError(f"Unexpected backend probe response: {probe_response}")

    message = str(probe_response.get("message", ""))
    if "Missing LLM configuration" in message or "LLM request failed" in message:
        raise VerificationError(f"Configured LLM is not usable: {probe_response}")


def _assert_injected_plan_rejection() -> None:
    original_generate_actions = planner_module.generate_actions
    try:
        planner_module.generate_actions = lambda prompt, scene_state: [  # type: ignore[assignment]
            {
                "action_id": "create_cube",
                "tool": "mesh.create_primitive",
                "params": {
                    "primitive_type": "cube",
                    "location": [0, 0, 0],
                    "bogus": 1,
                },
            }
        ]
        result = planner_module.plan("create cube", {})
    finally:
        planner_module.generate_actions = original_generate_actions  # type: ignore[assignment]

    if result.status != "error":
        raise VerificationError(f"Injected invalid plan was not rejected: {result}")
    if result.actions != []:
        raise VerificationError(f"Injected invalid plan unexpectedly produced actions: {result}")


def _build_blender_validation_script(zip_path: Path, repo_root: Path, host: str, port: int) -> str:
    return f"""
from __future__ import annotations

import json
import math
import time

import addon_utils
import bpy

ZIP_PATH = r"{zip_path}"
REPO_ROOT = r"{repo_root}"
BASE_URL = "http://{host}:{port}"


def assert_close_vector(actual, expected, label):
    actual_rounded = tuple(round(float(value), 4) for value in actual)
    expected_rounded = tuple(round(float(value), 4) for value in expected)
    if actual_rounded != expected_rounded:
        raise AssertionError(f"{{label}} mismatch: expected {{expected_rounded}}, got {{actual_rounded}}")


def clear_scene():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.context.view_layer.update()


def ensure_single_cube():
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "Cube"
    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.context.view_layer.update()
    return cube


def install_and_reload_addon():
    global create_task

    try:
        bpy.ops.preferences.addon_disable(module="vectra")
    except Exception:
        pass
    install_result = bpy.ops.preferences.addon_install(filepath=ZIP_PATH, overwrite=True)
    enable_result = bpy.ops.preferences.addon_enable(module="vectra")
    print("ADDON_INSTALL", install_result)
    print("ADDON_ENABLE", enable_result)
    default_loaded, state_loaded = addon_utils.check("vectra")
    print("ADDON_CHECK", default_loaded, state_loaded)
    assert state_loaded

    import vectra
    from vectra.bridge.client import create_task as bridge_create_task

    prefs = bpy.context.preferences.addons["vectra"].preferences
    prefs.dev_source_path = REPO_ROOT
    reload_result = bpy.ops.vectra.reload_dev()
    print("RELOAD_DEV", reload_result)
    status = vectra.addon_loader.get_runtime_status()
    print("RUNTIME_STATUS", status)
    assert status.mode == "dev-source"
    assert status.source_path == REPO_ROOT
    create_task = bridge_create_task


def run_operator_prompt(prompt):
    import vectra.operators.run_task as run_task_module

    scene = bpy.context.scene
    scene.vectra_prompt = prompt
    result = bpy.ops.vectra.run_task()
    print("OPERATOR_RESULT", prompt, result)
    deadline = time.time() + 30.0
    while scene.vectra_request_in_flight and time.time() < deadline:
        run_task_module._poll_request_result()
        time.sleep(0.1)
    if scene.vectra_request_in_flight:
        raise AssertionError(f"Timed out waiting for prompt: {{prompt}}")
    return scene.vectra_phase, scene.vectra_status


def snapshot_scene_state():
    import vectra.operators.run_task as run_task_module

    return run_task_module._build_scene_state(bpy.context)


def assert_mesh_at_origin_after_create():
    clear_scene()
    phase, status = run_operator_prompt("create a cube")
    if phase != "success":
        raise AssertionError(f"create a cube failed: {{phase}} / {{status}}")
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(mesh_objects) != 1:
        raise AssertionError(f"Expected exactly one mesh object, found {{len(mesh_objects)}}")
    assert_close_vector(mesh_objects[0].location, (0.0, 0.0, 0.0), "create cube location")


def assert_move(prompt, expected_location):
    cube = ensure_single_cube()
    phase, status = run_operator_prompt(prompt)
    if phase != "success":
        raise AssertionError(f"{{prompt}} failed: {{phase}} / {{status}}")
    assert_close_vector(cube.location, expected_location, prompt)


def assert_rotate():
    cube = ensure_single_cube()
    phase, status = run_operator_prompt("rotate cube 45 degrees")
    if phase != "success":
        raise AssertionError(f"rotate cube 45 degrees failed: {{phase}} / {{status}}")
    expected = round(math.radians(45.0), 4)
    actual = round(float(cube.rotation_euler[2]), 4)
    if actual != expected:
        raise AssertionError(f"Unexpected Z rotation: expected {{expected}}, got {{actual}}")
    assert_close_vector((cube.rotation_euler[0], cube.rotation_euler[1]), (0.0, 0.0), "rotation other axes")


def assert_invalid_prompt_fails():
    cube = ensure_single_cube()
    phase, status = run_operator_prompt("move cube somewhere weird")
    if phase != "error":
        raise AssertionError(f"Expected error phase, got {{phase}} / {{status}}")
    if not status.startswith("No actions returned:"):
        raise AssertionError(f"Unexpected failure status: {{status}}")
    assert_close_vector(cube.location, (0.0, 0.0, 0.0), "invalid prompt location")


def assert_repeat_is_deterministic():
    ensure_single_cube()
    payload = {{
        "prompt": "move cube forward 2",
        "scene_state": snapshot_scene_state(),
        "images": [],
    }}
    first = create_task(payload, base_url=BASE_URL, timeout=20.0)
    second = create_task(payload, base_url=BASE_URL, timeout=20.0)
    print("DIRECT_REPEAT_FIRST", json.dumps(first, sort_keys=True))
    print("DIRECT_REPEAT_SECOND", json.dumps(second, sort_keys=True))
    if first != second:
        raise AssertionError(f"Repeated backend response mismatch: {{first}} != {{second}}")

    for run_number in (1, 2):
        cube = ensure_single_cube()
        phase, status = run_operator_prompt("move cube forward 2")
        if phase != "success":
            raise AssertionError(f"repeat run {{run_number}} failed: {{phase}} / {{status}}")
        assert_close_vector(cube.location, (0.0, 2.0, 0.0), f"repeat run {{run_number}} location")


install_and_reload_addon()
assert_mesh_at_origin_after_create()
assert_move("move cube forward 2", (0.0, 2.0, 0.0))
assert_move("move cube back 2", (0.0, -2.0, 0.0))
assert_move("move cube up 3", (0.0, 0.0, 3.0))
assert_move("move cube down 1", (0.0, 0.0, -1.0))
assert_rotate()
assert_invalid_prompt_fails()
assert_repeat_is_deterministic()
print("BLENDER_PHASE1_VERIFICATION", "ok")
"""


def _run_blender_validation(blender_bin: Path, zip_path: Path, host: str, port: int) -> str:
    if not blender_bin.exists():
        raise VerificationError(f"Blender executable not found: {blender_bin}")

    with tempfile.TemporaryDirectory(prefix="vectra-phase1-") as temp_dir:
        script_path = Path(temp_dir) / "verify_phase1_blender.py"
        script_path.write_text(
            _build_blender_validation_script(zip_path, REPO_ROOT, host, port),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(blender_bin),
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(script_path),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    combined_output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise VerificationError(
            f"Blender validation failed with exit code {completed.returncode}.\n{combined_output}"
        )
    return combined_output


def _assert_log_visibility(backend_output: str, blender_output: str) -> None:
    required_backend_markers = [
        "planner_prompt",
        "planner_scene_state",
        "llm_raw_output",
        "llm_parsed_json",
        "planner_validated_actions",
    ]
    for marker in required_backend_markers:
        if marker not in backend_output:
            raise VerificationError(f"Backend logs missing marker '{marker}'")

    if "execution_report" not in blender_output:
        raise VerificationError("Blender logs missing marker 'execution_report'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vectra Phase 1 live verification")
    parser.add_argument(
        "--python-bin",
        default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python executable to use for backend and packaging",
    )
    parser.add_argument(
        "--blender-bin",
        default=DEFAULT_BLENDER_BIN,
        help="Path to the Blender executable",
    )
    parser.add_argument(
        "--backend-host",
        default=DEFAULT_BACKEND_HOST,
        help="Backend host for uvicorn",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help="Backend port for uvicorn",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_bin = Path(args.python_bin).expanduser().resolve()
    blender_bin = Path(args.blender_bin).expanduser().resolve()

    _check_python_environment(python_bin)
    zip_path = _build_addon_archive(python_bin)
    backend, backend_lines = _start_backend(
        python_bin=python_bin,
        host=args.backend_host,
        port=args.backend_port,
    )
    backend_output = ""
    try:
        health = health_check(base_url=f"http://{args.backend_host}:{args.backend_port}", timeout=2.0)
        if health != {"status": "ok"}:
            raise VerificationError(f"Unexpected health response: {health}")

        _assert_live_backend_round_trip(f"http://{args.backend_host}:{args.backend_port}")
        _assert_injected_plan_rejection()
        blender_output = _run_blender_validation(
            blender_bin,
            zip_path,
            args.backend_host,
            args.backend_port,
        )
        time.sleep(0.5)
        backend_output = "".join(backend_lines)
        _assert_log_visibility(backend_output, blender_output)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "backend_health": health,
                    "addon_zip": str(zip_path),
                    "blender_verification": "passed",
                },
                indent=2,
            )
        )
        return 0
    except (BridgeClientError, VerificationError) as exc:
        backend_output = backend_output or "".join(backend_lines)
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "backend_logs": backend_output[-4000:],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        _stop_backend(backend)


if __name__ == "__main__":
    raise SystemExit(main())
