from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TASK_TIMEOUT_SECONDS = 30.0


class BridgeClientError(Exception):
    """Base exception for Vectra bridge client errors."""


class BridgeConnectionError(BridgeClientError):
    """Raised when the backend cannot be reached."""


class BridgeTimeoutError(BridgeClientError):
    """Raised when the backend does not respond in time."""


class BridgeResponseError(BridgeClientError):
    """Raised when the backend returns an invalid response."""


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _format_connection_reason(reason: object) -> str:
    if isinstance(reason, socket.timeout):
        return "timed out"
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused"
    if isinstance(reason, OSError):
        detail = getattr(reason, "strerror", None)
        if detail:
            return str(detail)
    return str(reason)


def _decode_json_response(response: Any) -> dict[str, Any]:
    try:
        data = json.loads(response.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BridgeResponseError("Backend returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise BridgeResponseError("Backend response must be a JSON object")

    return data


def _handle_url_error(exc: error.URLError, *, url: str) -> None:
    reason = exc.reason
    if isinstance(reason, socket.timeout):
        raise BridgeTimeoutError(f"Backend request timed out for {url}") from exc
    raise BridgeConnectionError(
        f"Failed to connect to backend at {url}: {_format_connection_reason(reason)}"
    ) from exc


def _request_json(
    method: str,
    path: str,
    *,
    base_url: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    encoded_payload = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        encoded_payload = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(
        url=f"{_normalize_base_url(base_url)}{path}",
        data=encoded_payload,
        headers=headers,
        method=method,
    )
    request_url = req.full_url

    try:
        with request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise BridgeResponseError(f"Backend returned HTTP {response.status}")
            return _decode_json_response(response)
    except error.HTTPError as exc:
        raise BridgeResponseError(f"Backend returned HTTP {exc.code}") from exc
    except error.URLError as exc:
        _handle_url_error(exc, url=request_url)
    except TimeoutError as exc:
        raise BridgeTimeoutError(f"Backend request timed out for {request_url}") from exc


def health_check(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.0) -> dict[str, Any]:
    data = _request_json("GET", "/health", base_url=base_url, timeout=timeout)
    if data.get("status") != "ok":
        raise BridgeResponseError("Health response missing status='ok'")
    return data


def create_task(
    payload: dict[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    data = _request_json(
        "POST",
        "/task/create",
        base_url=base_url,
        timeout=timeout,
        payload=payload,
    )

    required_keys = {"status", "message", "actions"}
    missing = required_keys.difference(data)
    if missing:
        raise BridgeResponseError(f"Task response missing keys: {sorted(missing)}")
    if not isinstance(data["actions"], list):
        raise BridgeResponseError("Task response field 'actions' must be a list")
    return data


def create_agent_step(
    payload: dict[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    data = _request_json(
        "POST",
        "/agent/step",
        base_url=base_url,
        timeout=timeout,
        payload=payload,
    )

    required_keys = {
        "status",
        "message",
        "narration",
        "understanding",
        "plan",
        "intended_actions",
        "preferred_execution_mode",
        "continue_loop",
        "question",
        "expected_outcome",
        "execution",
    }
    missing = required_keys.difference(data)
    if missing:
        raise BridgeResponseError(f"Agent step response missing keys: {sorted(missing)}")
    if not isinstance(data["execution"], dict):
        raise BridgeResponseError("Agent step response field 'execution' must be an object")
    return data
