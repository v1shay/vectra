from __future__ import annotations

from typing import Any

try:
    from .models import TaskCreateResponse
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from models import TaskCreateResponse


def build_task_response(
    prompt: str,
    scene_state: dict[str, Any],
    images: list[Any],
) -> TaskCreateResponse:
    del prompt
    del scene_state
    del images
    return TaskCreateResponse(status="ok", message="received", actions=[])
