from __future__ import annotations

from pathlib import Path

import pytest

import vectra.runtime_service as runtime_service
from vectra.bridge.client import BridgeConnectionError


@pytest.fixture(autouse=True)
def _reset_runtime_service_state() -> None:
    runtime_service.reset_managed_backend_state()
    yield
    runtime_service.reset_managed_backend_state()


def test_ensure_local_backend_returns_when_health_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(
        runtime_service,
        "health_check",
        lambda **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        runtime_service,
        "_start_backend_process",
        lambda repo_root, base_url: started.append(f"{repo_root}:{base_url}"),
    )

    runtime_service.ensure_local_backend(base_url="http://127.0.0.1:8000")

    assert started == []


def test_ensure_local_backend_starts_repo_runtime_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    health_attempts = {"count": 0}

    def fake_health_check(*, base_url: str, timeout: float) -> dict[str, str]:
        del timeout
        health_attempts["count"] += 1
        if health_attempts["count"] == 1:
            raise BridgeConnectionError(f"Failed to connect to backend at {base_url}/health: connection refused")
        return {"status": "ok"}

    monkeypatch.setattr(runtime_service, "health_check", fake_health_check)
    monkeypatch.setattr(
        runtime_service,
        "_discover_repo_root",
        lambda repo_root_hint: Path("/tmp/vectra"),
    )
    monkeypatch.setattr(
        runtime_service,
        "_start_backend_process",
        lambda repo_root, base_url: calls.append((repo_root, base_url)),
    )

    runtime_service.ensure_local_backend(
        base_url="http://127.0.0.1:8000",
        repo_root_hint="/tmp/vectra",
        startup_timeout=0.1,
    )

    assert calls == [(Path("/tmp/vectra"), "http://127.0.0.1:8000")]


def test_ensure_local_backend_reports_actionable_error_when_repo_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_service,
        "health_check",
        lambda **kwargs: (_ for _ in ()).throw(
            BridgeConnectionError("Failed to connect to backend at http://127.0.0.1:8000/health: connection refused")
        ),
    )
    monkeypatch.setattr(runtime_service, "_discover_repo_root", lambda repo_root_hint: None)

    with pytest.raises(BridgeConnectionError, match="could not find a repo checkout with agent_runtime"):
        runtime_service.ensure_local_backend(
            base_url="http://127.0.0.1:8000",
            repo_root_hint="/tmp/missing-vectra",
        )


def test_manual_start_command_uses_repo_root_package_entrypoint() -> None:
    command = runtime_service._manual_start_command("/tmp/vectra", "http://127.0.0.1:8000")

    assert "cd /tmp/vectra &&" in command
    assert "uvicorn agent_runtime.main:app --reload" in command


def test_start_backend_process_uses_package_entrypoint_from_repo_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "vectra"
    repo_root.mkdir()
    venv_bin = repo_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_bin = venv_bin / "python"
    python_bin.write_text("", encoding="utf-8")

    launched: dict[str, object] = {}

    class DummyProcess:
        def poll(self) -> None:
            return None

    def fake_popen(command, cwd, env, stdout, stderr, text, start_new_session):
        launched["command"] = command
        launched["cwd"] = cwd
        launched["env"] = env
        launched["stdout"] = stdout
        launched["stderr"] = stderr
        launched["text"] = text
        launched["start_new_session"] = start_new_session
        return DummyProcess()

    monkeypatch.setattr(runtime_service.subprocess, "Popen", fake_popen)

    runtime_service._start_backend_process(repo_root, "http://127.0.0.1:8000")

    assert launched["cwd"] == repo_root
    assert launched["command"] == [
        str(python_bin),
        "-m",
        "uvicorn",
        "agent_runtime.main:app",
        "--reload",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
