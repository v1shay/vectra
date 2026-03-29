from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from agent_runtime.main import app
from blender_addon.bridge.client import (
    BridgeConnectionError,
    create_task,
    health_check,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_server() -> str:
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
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
    assert create_task(payload, base_url=live_server) == {
        "status": "ok",
        "message": "received",
        "actions": [],
    }


def test_bridge_client_raises_clear_error_when_backend_is_offline() -> None:
    unused_port = _find_free_port()
    base_url = f"http://127.0.0.1:{unused_port}"

    with pytest.raises(BridgeConnectionError):
        health_check(base_url=base_url, timeout=0.2)
