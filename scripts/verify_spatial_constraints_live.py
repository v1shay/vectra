from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Vectra spatial-constraint live Blender validation.")
    parser.add_argument("--blender-path", default=DEFAULT_BLENDER)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    blender_path = Path(args.blender_path).expanduser()
    if not blender_path.exists():
        print(json.dumps({"status": "skipped", "reason": f"Blender executable not found: {blender_path}"}))
        return 2

    env = os.environ.copy()
    env["VECTRA_DIRECTOR_AUDIT_MODE"] = "1"
    if not args.allow_fallback:
        env["VECTRA_DISABLE_PROVIDER_FALLBACK"] = "1"

    command = [
        sys.executable,
        "-m",
        "vectra.audit.runner",
        "--launch-backend",
        "--modes",
        "vectra-dev",
        "--limit",
        str(args.limit),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--blender-path",
        str(blender_path),
    ]
    process = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    if process.stdout:
        print(process.stdout)
    if process.returncode != 0:
        print(process.stderr, file=sys.stderr)
    latest_summary = _latest_summary()
    if latest_summary is not None:
        print(json.dumps(latest_summary, indent=2))
    return process.returncode


def _latest_summary() -> dict[str, object] | None:
    audits_root = REPO_ROOT / ".vectra" / "audits"
    if not audits_root.exists():
        return None
    audit_dirs = sorted(path for path in audits_root.iterdir() if path.is_dir())
    if not audit_dirs:
        return None
    latest = audit_dirs[-1]
    summary_path = latest / "summary.json"
    artifacts_path = latest / "artifacts.json"
    payload: dict[str, object] = {"audit_dir": str(latest)}
    if summary_path.exists():
        try:
            payload["summary"] = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            payload["summary_path"] = str(summary_path)
    if artifacts_path.exists():
        payload["artifacts_path"] = str(artifacts_path)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
