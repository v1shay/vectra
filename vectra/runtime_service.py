from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

from .bridge.client import BridgeClientError, BridgeConnectionError, health_check

DEFAULT_BACKEND_STARTUP_TIMEOUT_SECONDS = 15.0
HEALTH_POLL_INTERVAL_SECONDS = 0.1

_MANAGED_BACKEND_PROCESS: subprocess.Popen[str] | None = None
_MANAGED_BACKEND_LOG_HANDLE: TextIO | None = None
_MANAGED_BACKEND_LOG_PATH: Path | None = None


def _is_managed_backend_alive() -> bool:
    return _MANAGED_BACKEND_PROCESS is not None and _MANAGED_BACKEND_PROCESS.poll() is None


def _close_managed_log_handle_if_inactive() -> None:
    global _MANAGED_BACKEND_LOG_HANDLE

    if _is_managed_backend_alive():
        return
    if _MANAGED_BACKEND_LOG_HANDLE is None:
        return

    _MANAGED_BACKEND_LOG_HANDLE.close()
    _MANAGED_BACKEND_LOG_HANDLE = None


def reset_managed_backend_state(*, stop_process: bool = False) -> None:
    global _MANAGED_BACKEND_PROCESS, _MANAGED_BACKEND_LOG_HANDLE, _MANAGED_BACKEND_LOG_PATH

    if stop_process and _is_managed_backend_alive():
        _MANAGED_BACKEND_PROCESS.terminate()
        try:
            _MANAGED_BACKEND_PROCESS.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _MANAGED_BACKEND_PROCESS.kill()
            _MANAGED_BACKEND_PROCESS.wait(timeout=5.0)

    _MANAGED_BACKEND_PROCESS = None
    _MANAGED_BACKEND_LOG_PATH = None
    _close_managed_log_handle_if_inactive()


def _local_backend_target(base_url: str) -> tuple[str, int] | None:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    if parsed.scheme not in {"http", "https"}:
        return None
    if host not in {"127.0.0.1", "localhost"}:
        return None

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _candidate_repo_roots(repo_root_hint: str | None) -> list[Path]:
    candidates: list[Path] = []
    raw_candidates = []
    if repo_root_hint:
        raw_candidates.append(Path(repo_root_hint).expanduser())
    raw_candidates.append(Path(__file__).resolve().parents[1])

    for candidate in raw_candidates:
        resolved = candidate.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def _is_valid_repo_root(candidate: Path) -> bool:
    return (candidate / "agent_runtime" / "main.py").is_file() and (candidate / "vectra").is_dir()


def _discover_repo_root(repo_root_hint: str | None) -> Path | None:
    for candidate in _candidate_repo_roots(repo_root_hint):
        if _is_valid_repo_root(candidate):
            return candidate
    return None


def _repo_python_bin(repo_root: Path) -> Path | None:
    candidates = (
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _manual_start_command(repo_root_hint: str | None, base_url: str) -> str:
    local_target = _local_backend_target(base_url) or ("127.0.0.1", 8000)
    host, port = local_target
    if repo_root_hint:
        repo_root = Path(repo_root_hint).expanduser()
        python_bin = _repo_python_bin(repo_root) or repo_root / ".venv" / "bin" / "python"
        agent_runtime_dir = repo_root / "agent_runtime"
        return (
            f"cd {shlex.quote(str(agent_runtime_dir))} && "
            f"{shlex.quote(str(python_bin))} -m uvicorn main:app --reload "
            f"--host {host} --port {port}"
        )

    return (
        "cd <repo>/agent_runtime && "
        "<repo>/.venv/bin/python -m uvicorn main:app --reload "
        f"--host {host} --port {port}"
    )


def _backend_log_path(repo_root: Path) -> Path:
    log_dir = repo_root / ".vectra"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "backend.log"


def _start_backend_process(repo_root: Path, base_url: str) -> None:
    global _MANAGED_BACKEND_PROCESS, _MANAGED_BACKEND_LOG_HANDLE, _MANAGED_BACKEND_LOG_PATH

    local_target = _local_backend_target(base_url)
    if local_target is None:
        raise BridgeConnectionError(
            f"Backend is offline at {base_url}. Auto-start is only supported for local backend URLs."
        )

    python_bin = _repo_python_bin(repo_root)
    if python_bin is None:
        raise BridgeConnectionError(
            "Backend is offline and Vectra could not auto-start it because the repo virtualenv "
            f"is missing at {repo_root / '.venv'}. Start the backend manually with: "
            f"{_manual_start_command(str(repo_root), base_url)}"
        )

    _close_managed_log_handle_if_inactive()
    log_path = _backend_log_path(repo_root)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"\n=== Starting Vectra backend at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    )
    log_handle.flush()

    host, port = local_target
    command = [
        str(python_bin),
        "-m",
        "uvicorn",
        "main:app",
        "--reload",
        "--host",
        host,
        "--port",
        str(port),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root / "agent_runtime",
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        raise BridgeConnectionError(
            "Backend is offline and Vectra could not launch the runtime. "
            f"Command: {_manual_start_command(str(repo_root), base_url)}. Error: {exc}"
        ) from exc

    _MANAGED_BACKEND_PROCESS = process
    _MANAGED_BACKEND_LOG_HANDLE = log_handle
    _MANAGED_BACKEND_LOG_PATH = log_path


def _wait_for_backend_health(base_url: str, startup_timeout: float) -> None:
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _MANAGED_BACKEND_PROCESS is not None and _MANAGED_BACKEND_PROCESS.poll() is not None:
            exit_code = _MANAGED_BACKEND_PROCESS.returncode
            raise BridgeConnectionError(
                "Backend is offline and the auto-started runtime exited before it became healthy. "
                f"Exit code: {exit_code}. Log: {_MANAGED_BACKEND_LOG_PATH}"
            )

        try:
            health_check(base_url=base_url, timeout=0.5)
            return
        except BridgeConnectionError:
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
        except BridgeClientError:
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)

    raise BridgeConnectionError(
        "Backend is offline and the auto-started runtime did not become healthy in time. "
        f"Log: {_MANAGED_BACKEND_LOG_PATH}"
    )


def ensure_local_backend(
    *,
    base_url: str,
    repo_root_hint: str | None = None,
    startup_timeout: float = DEFAULT_BACKEND_STARTUP_TIMEOUT_SECONDS,
) -> None:
    try:
        health_check(base_url=base_url, timeout=0.5)
        return
    except BridgeConnectionError as exc:
        initial_error = exc
    except BridgeClientError:
        return

    if _is_managed_backend_alive():
        _wait_for_backend_health(base_url, startup_timeout)
        return

    repo_root = _discover_repo_root(repo_root_hint)
    if repo_root is None:
        raise BridgeConnectionError(
            f"{initial_error}. Vectra could not auto-start the backend because it could not find "
            "a repo checkout with agent_runtime/. Set the Vectra Development Source Path to the repo root "
            f"or start the backend manually with: {_manual_start_command(repo_root_hint, base_url)}"
        ) from initial_error

    _start_backend_process(repo_root, base_url)
    _wait_for_backend_health(base_url, startup_timeout)
