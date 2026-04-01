from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vectra.bridge.client import health_check

DEFAULT_BLENDER_BIN = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_BACKEND_HOST = "127.0.0.1"


class AgentModeValidationError(RuntimeError):
    """Raised when experimental agent mode validation fails."""


def _repo_python() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.is_file():
        return candidate
    raise AgentModeValidationError(f"Repo virtualenv python was not found at {candidate}")


def _build_addon_archive(python_bin: Path) -> Path:
    completed = subprocess.run(
        [str(python_bin), str(REPO_ROOT / "scripts" / "package_addon.py")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AgentModeValidationError(
            f"Failed to build add-on archive:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    zip_path = REPO_ROOT / "vectra_addon.zip"
    if not zip_path.exists():
        raise AgentModeValidationError(f"Packaged add-on archive not found: {zip_path}")
    return zip_path


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_BACKEND_HOST, 0))
        return int(sock.getsockname()[1])


def _read_process_output(process: subprocess.Popen[str], sink: list[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        sink.append(line)


def _start_backend(python_bin: Path, host: str, port: int) -> tuple[subprocess.Popen[str], list[str]]:
    backend_env = os.environ.copy()
    backend_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [
            str(python_bin),
            "-m",
            "uvicorn",
            "main:app",
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

    output: list[str] = []
    reader = threading.Thread(
        target=_read_process_output,
        args=(process, output),
        daemon=True,
        name="vectra-agent-backend-reader",
    )
    reader.start()

    base_url = f"http://{host}:{port}"
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if process.poll() is not None:
            raise AgentModeValidationError(
                f"Backend exited before becoming healthy.\n{''.join(output)}"
            )
        try:
            if health_check(base_url=base_url, timeout=0.5) == {"status": "ok"}:
                return process, output
        except Exception:
            time.sleep(0.1)

    raise AgentModeValidationError(f"Timed out waiting for backend health.\n{''.join(output)}")


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


def _build_blender_validation_script(zip_path: Path, base_url: str) -> str:
    return textwrap.dedent(
        f"""
        import time
        import bpy
        import addon_utils

        zip_path = r"{zip_path}"
        base_url = r"{base_url}"

        install_result = bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
        print("ADDON_INSTALL", install_result)
        enable_result = bpy.ops.preferences.addon_enable(module="vectra")
        print("ADDON_ENABLE", enable_result)

        default_loaded, state_loaded = addon_utils.check("vectra")
        print("ADDON_CHECK", default_loaded, state_loaded)
        assert state_loaded

        import vectra.operators.run_task as run_task_module

        def clear_mesh_objects():
            for obj in list(bpy.data.objects):
                if obj.type == "MESH":
                    bpy.data.objects.remove(obj, do_unlink=True)

        def mesh_objects():
            return [obj for obj in bpy.data.objects if obj.type == "MESH"]

        def run_prompt(prompt: str, mode: str):
            scene = bpy.context.scene
            scene.vectra_prompt = prompt
            scene.vectra_execution_mode = mode
            result = bpy.ops.vectra.run_task()
            print("RUN_PROMPT", mode, prompt, result)

            deadline = time.time() + 30.0
            while scene.vectra_request_in_flight and time.time() < deadline:
                run_task_module._poll_request_result()
                time.sleep(0.05)

            print("FINAL_STATE", mode, prompt, scene.vectra_phase, scene.vectra_status, scene.vectra_iteration)
            print("TRANSCRIPT", mode, prompt, scene.vectra_agent_transcript.replace("\\n", " | "))
            assert scene.vectra_phase == "success", (mode, prompt, scene.vectra_status, scene.vectra_agent_transcript)
            assert scene.vectra_agent_transcript.strip(), (mode, prompt, "empty transcript")
            assert scene.vectra_pending_question == "", (mode, prompt, scene.vectra_pending_question)
            return scene.vectra_iteration

        for mode in ("vectra-dev", "vectra-code"):
            clear_mesh_objects()
            iterations = run_prompt("make two cubes", mode)
            cubes = mesh_objects()
            assert len(cubes) >= 2, (mode, "two cubes", [obj.name for obj in cubes])
            assert iterations >= 2, (mode, "two cubes iterations", iterations)

            run_prompt("move them apart", mode)
            cubes = sorted(mesh_objects(), key=lambda obj: obj.name)
            assert len(cubes) >= 2, (mode, "move apart cubes", [obj.name for obj in cubes])
            spacing = abs(float(cubes[1].location[0]) - float(cubes[0].location[0]))
            assert spacing >= 2.5, (mode, "move apart spacing", spacing)

            clear_mesh_objects()
            iterations = run_prompt("create a staircase of cubes", mode)
            stairs = sorted(mesh_objects(), key=lambda obj: obj.name)
            assert len(stairs) >= 5, (mode, "stair count", [obj.name for obj in stairs])
            assert iterations >= 5, (mode, "stair iterations", iterations)

            clear_mesh_objects()
            iterations = run_prompt("put a cube on top of another", mode)
            stacked = sorted(mesh_objects(), key=lambda obj: obj.location[2])
            assert len(stacked) >= 2, (mode, "stacked cubes", [obj.name for obj in stacked])
            height_gap = float(stacked[1].location[2]) - float(stacked[0].location[2])
            assert height_gap >= 1.9, (mode, "stack height", height_gap)
            assert iterations >= 2, (mode, "stack iterations", iterations)

        print("AGENT_MODE_VALIDATION", "passed", base_url)
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VECTRA_AGENT_MODE end-to-end in Blender")
    parser.add_argument(
        "--blender-bin",
        default=DEFAULT_BLENDER_BIN,
        help="Path to the Blender executable",
    )
    args = parser.parse_args()

    python_bin = _repo_python()
    blender_bin = Path(args.blender_bin).expanduser().resolve()
    if not blender_bin.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_bin}")

    zip_path = _build_addon_archive(python_bin)
    port = _find_free_port()
    base_url = f"http://{DEFAULT_BACKEND_HOST}:{port}"
    backend_process, _backend_output = _start_backend(python_bin, DEFAULT_BACKEND_HOST, port)

    temp_root = Path(tempfile.mkdtemp(prefix="vectra-agent-mode-"))
    script_path = temp_root / "validate_agent_mode.py"
    script_path.write_text(_build_blender_validation_script(zip_path, base_url), encoding="utf-8")

    env = os.environ.copy()
    env["BLENDER_USER_CONFIG"] = str(temp_root / "config")
    env["BLENDER_USER_SCRIPTS"] = str(temp_root / "scripts")
    env["BLENDER_USER_DATAFILES"] = str(temp_root / "data")
    env["VECTRA_AGENT_MODE"] = "true"
    env["VECTRA_AGENT_MEMORY_ENABLED"] = "false"
    env["VECTRA_BASE_URL"] = base_url

    try:
        completed = subprocess.run(
            [
                str(blender_bin),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(script_path),
            ],
            env=env,
            check=False,
            text=True,
        )
        return completed.returncode
    finally:
        _stop_backend(backend_process)
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
