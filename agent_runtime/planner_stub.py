from __future__ import annotations

from typing import Any

from .models import TaskCreateResponse


def build_task_response(
    prompt: str,
    scene_state: dict[str, Any],
    images: list[Any],
) -> TaskCreateResponse:
    del prompt
    del scene_state
    del images
    return TaskCreateResponse(status="ok", message="received", actions=[])
