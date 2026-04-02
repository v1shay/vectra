from __future__ import annotations

import socket
import threading
import time
from typing import Iterator

import pytest
import uvicorn

import agent_runtime.main as runtime_main
from agent_runtime.planner import PlannerResult
from tests.action_fixtures import CREATE_CUBE_ACTIONS
from vectra.bridge.client import (
    BridgeConnectionError,
    create_task,
    health_check,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(
        runtime_main,
        "plan",
        lambda prompt, scene_state: PlannerResult(
            status="ok",
            actions=CREATE_CUBE_ACTIONS,
            message=f"planned for {prompt}:{scene_state.get('current_frame', 'missing')}",
        ),
    )
    port = _find_free_port()
    config = uvicorn.Config(runtime_main.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if getattr(server, "started", False):
            try:
                health_check(base_url=base_url, timeout=0.5)
                break
            except Exception:
                pass
        time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5.0)
        pytest.fail("Timed out waiting for uvicorn test server to start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5.0)


def test_bridge_client_talks_to_live_backend(live_server: str) -> None:
    payload = {
        "prompt": "Add a light",
        "scene_state": {
            "active_object": None,
            "selected_objects": [],
            "current_frame": 1,
        },
        "images": [],
    }

    assert health_check(base_url=live_server) == {"status": "ok"}
    response = create_task(payload, base_url=live_server)

    assert response["status"] == "ok"
    assert response["message"] == "planned for Add a light:1"
    assert response["actions"] == CREATE_CUBE_ACTIONS
    assert response["assumptions"] == []
    assert response["metadata"] == {}


def test_bridge_client_raises_clear_error_when_backend_is_offline() -> None:
    unused_port = _find_free_port()
    base_url = f"http://127.0.0.1:{unused_port}"

    with pytest.raises(BridgeConnectionError, match="Failed to connect to backend at"):
        health_check(base_url=base_url, timeout=0.2)
