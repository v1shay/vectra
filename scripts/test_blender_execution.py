from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build_blender_validation_script(zip_path: Path) -> str:
    return f"""
import time
import bpy
import addon_utils

zip_path = r"{zip_path}"

install_result = bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
print("ADDON_INSTALL", install_result)
enable_result = bpy.ops.preferences.addon_enable(module="vectra")
print("ADDON_ENABLE", enable_result)

default_loaded, state_loaded = addon_utils.check("vectra")
print("ADDON_CHECK", default_loaded, state_loaded)
assert state_loaded

scene = bpy.context.scene
scene.vectra_prompt = "scripted blender execution test"
result = bpy.ops.vectra.run_task()
print("OPERATOR_RESULT", result)

import vectra.operators.run_task as run_task_module

deadline = time.time() + 10.0
while scene.vectra_request_in_flight and time.time() < deadline:
    run_task_module._poll_request_result()
    time.sleep(0.1)

obj = bpy.data.objects.get("VectraCube")
print("FINAL_STATUS", scene.vectra_status)
print("FINAL_PHASE", scene.vectra_phase)
print("OBJECT_FOUND", obj is not None)
if obj is not None:
    print("OBJECT_LOCATION", tuple(round(v, 4) for v in obj.location))

assert scene.vectra_phase == "success"
assert scene.vectra_status == "Executed 2 action(s) successfully"
assert obj is not None
assert tuple(round(v, 4) for v in obj.location) == (2.0, 0.0, 0.0)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the packaged Vectra add-on headlessly in Blender")
    parser.add_argument(
        "--blender-bin",
        default="/Applications/Blender.app/Contents/MacOS/Blender",
        help="Path to the Blender executable",
    )
    parser.add_argument(
        "--zip-path",
        default=str(Path(__file__).resolve().parents[1] / "vectra_addon.zip"),
        help="Path to vectra_addon.zip",
    )
    args = parser.parse_args()

    blender_bin = Path(args.blender_bin).expanduser().resolve()
    zip_path = Path(args.zip_path).expanduser().resolve()
    if not blender_bin.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_bin}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Add-on zip not found: {zip_path}")

    temp_root = Path(tempfile.mkdtemp(prefix="vectra-blender-exec-"))
    script_path = temp_root / "validate_blender_execution.py"
    script_path.write_text(build_blender_validation_script(zip_path), encoding="utf-8")

    env = os.environ.copy()
    env["BLENDER_USER_CONFIG"] = str(temp_root / "config")
    env["BLENDER_USER_SCRIPTS"] = str(temp_root / "scripts")
    env["BLENDER_USER_DATAFILES"] = str(temp_root / "data")

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
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
