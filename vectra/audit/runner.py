from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .corpus import AUDIT_CASES, DEFAULT_MODES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8001"
DEFAULT_AUDIT_DIR = REPO_ROOT / ".vectra" / "audits"
DEFAULT_TIMEOUT_SECONDS = 240.0


@dataclass(frozen=True)
class RankedBottleneck:
    tag: str
    failed_runs: int
    budget_failures: int
    lost_turns: int
    hard_failures: int


def _healthcheck(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url.rstrip('/')}/health", timeout=2.0) as response:
            return response.status == 200
    except OSError:
        return False


def _repo_python() -> str:
    return str(REPO_ROOT / ".venv" / "bin" / "python")


def _start_backend(base_url: str) -> subprocess.Popen[str]:
    host = "127.0.0.1"
    port = int(base_url.rsplit(":", 1)[-1])
    process = subprocess.Popen(
        [_repo_python(), "-m", "uvicorn", "agent_runtime.main:app", "--host", host, "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if _healthcheck(base_url):
            return process
        if process.poll() is not None:
            break
        time.sleep(0.25)
    output = ""
    if process.stdout is not None:
        output = process.stdout.read()
    raise RuntimeError(f"Failed to start backend at {base_url}. Output:\n{output}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_blender_case(blender_path: str, base_url: str, prompt: str, mode: str, setup_id: str, artifact_path: Path, timeout_seconds: float) -> dict[str, Any]:
    config = {
        "prompt": prompt,
        "mode": mode,
        "setup_id": setup_id,
        "artifact_path": str(artifact_path),
        "base_url": base_url,
        "timeout_seconds": max(10.0, timeout_seconds - 5.0),
    }
    config_path = artifact_path.with_suffix(".config.json")
    _write_json(config_path, config)
    command = [
        blender_path,
        "--factory-startup",
        "--python",
        str(REPO_ROOT / "vectra" / "audit" / "blender_job.py"),
        "--",
        str(config_path),
    ]
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if process.returncode != 0:
        raise RuntimeError(f"Blender audit job failed for {mode}:{setup_id}:{prompt}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def _core_objective_achieved(artifact: dict[str, Any]) -> bool:
    quality = artifact.get("quality_flags", {})
    completion = artifact.get("completion_status", {})
    if completion.get("phase") == "success":
        return True
    if quality.get("object_count", 0) >= 3 and artifact.get("wasted_turns", 0) <= 1:
        return True
    return False


def _derive_tags(artifact: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    transcript = str(artifact.get("transcript", "")).lower()
    completion = artifact.get("completion_status", {})
    phase = str(completion.get("phase", "error"))
    status = str(completion.get("status", ""))
    timed_out = bool(artifact.get("timed_out", False))
    quality = artifact.get("quality_flags", {})
    prompt = str(artifact.get("prompt", "")).lower()
    turns_used = int(artifact.get("turns_used", 0))
    wasted_turns = int(artifact.get("wasted_turns", 0))
    batch_stats = artifact.get("batch_size_statistics", {})
    ratio = artifact.get("bulk_vs_micro_action_ratio", {})

    if "400 bad request" in transcript or "provider" in status.lower():
        tags.add("provider_transport")
    if "must be a 3-item list or tuple" in transcript or "validation" in transcript:
        tags.add("argument_robustness")
    if timed_out:
        tags.add("anti_stall_failure")
    if wasted_turns >= 2:
        tags.add("anti_stall_failure")
    if batch_stats.get("average", 0.0) < 1.5 and turns_used >= 4:
        tags.add("batching_inefficiency")
    if ratio.get("bulk_actions", 0) == 0 and ratio.get("micro_actions", 0) >= 4:
        tags.add("batching_inefficiency")
    if "animate" in prompt and not quality.get("has_animation"):
        tags.add("animation_quality")
    if any(token in prompt for token in ("scene", "room", "cinematic", "cool")):
        if not quality.get("has_light"):
            tags.add("scene_composition")
        if not quality.get("has_camera"):
            tags.add("scene_composition")
    if any(token in prompt for token in ("desk", "chair", "lamp")) and not quality.get("has_groups"):
        tags.add("composite_quality")
    if phase != "success":
        tags.add("under_budget_completion_failure")
    if phase != "success" and not _core_objective_achieved(artifact):
        tags.add("under_budget_completion_failure")
    if not tags and phase != "success":
        tags.add("tool_gap")
    return sorted(tags)


def _rank_bottlenecks(artifacts: list[dict[str, Any]]) -> list[RankedBottleneck]:
    counters: dict[str, dict[str, int]] = defaultdict(lambda: {"failed_runs": 0, "budget_failures": 0, "lost_turns": 0, "hard_failures": 0})
    for artifact in artifacts:
        tags = artifact.get("final_bottleneck_tags", [])
        failed = artifact.get("completion_status", {}).get("phase") != "success"
        wasted_turns = int(artifact.get("wasted_turns", 0))
        over_budget = failed or wasted_turns >= 2
        for tag in tags:
            counters[tag]["failed_runs"] += 1 if failed else 0
            counters[tag]["budget_failures"] += 1 if over_budget else 0
            counters[tag]["lost_turns"] += wasted_turns
            counters[tag]["hard_failures"] += 1 if failed else 0
    ranked = [
        RankedBottleneck(
            tag=tag,
            failed_runs=values["failed_runs"],
            budget_failures=values["budget_failures"],
            lost_turns=values["lost_turns"],
            hard_failures=values["hard_failures"],
        )
        for tag, values in counters.items()
    ]
    ranked.sort(key=lambda item: (-item.failed_runs, -item.budget_failures, -item.lost_turns, -item.hard_failures, item.tag))
    return ranked


def _summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    passes = sum(1 for artifact in artifacts if artifact.get("completion_status", {}).get("phase") == "success")
    turn_counts = [int(artifact.get("turns_used", 0)) for artifact in artifacts]
    return {
        "total_runs": len(artifacts),
        "pass_rate": (passes / float(len(artifacts))) if artifacts else 0.0,
        "average_turns": (sum(turn_counts) / float(len(turn_counts))) if turn_counts else 0.0,
        "provider_usage": Counter(chain for artifact in artifacts for chain in artifact.get("provider_chain", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Vectra's live Blender audit suite.")
    parser.add_argument("--base-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--blender-path", default=DEFAULT_BLENDER_PATH)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of audit cases for a smoke run")
    parser.add_argument("--launch-backend", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = DEFAULT_AUDIT_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_process: subprocess.Popen[str] | None = None
    if args.launch_backend and not _healthcheck(args.base_url):
        backend_process = _start_backend(args.base_url)

    try:
        artifacts: list[dict[str, Any]] = []
        cases = list(AUDIT_CASES[: args.limit or len(AUDIT_CASES)])
        for mode in args.modes:
            for case_index, case in enumerate(cases, start=1):
                if case.modes is not None and mode not in case.modes:
                    continue
                artifact_path = output_dir / f"{mode}-{case_index:02d}.json"
                artifact = _run_blender_case(
                    args.blender_path,
                    args.base_url,
                    case.prompt,
                    mode,
                    case.setup_id,
                    artifact_path,
                    args.timeout_seconds,
                )
                artifact["expectations"] = asdict(case)
                artifact["core_objective_achieved"] = _core_objective_achieved(artifact)
                artifact["final_bottleneck_tags"] = _derive_tags(artifact)
                _write_json(artifact_path, artifact)
                artifacts.append(artifact)

        ranking = [asdict(item) for item in _rank_bottlenecks(artifacts)]
        summary = _summary(artifacts)
        _write_json(output_dir / "summary.json", {"summary": summary, "ranking": ranking})
        _write_json(output_dir / "artifacts.json", {"artifacts": artifacts})
        return 0
    finally:
        if backend_process is not None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                backend_process.kill()
                backend_process.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
