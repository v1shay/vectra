from __future__ import annotations

from typing import Any

try:
    from .models import ActionModel, TaskCreateResponse
except ImportError:  # pragma: no cover - supports local module imports from agent_runtime/
    from models import ActionModel, TaskCreateResponse


def build_task_response(
    prompt: str,
    scene_state: dict[str, Any],
    images: list[Any],
) -> TaskCreateResponse:
    del prompt
    del scene_state
    del images
    return TaskCreateResponse(
        status="ok",
        message="planned",
        actions=[
            ActionModel(
                action_id="create_cube",
                tool="mesh.create_primitive",
                params={
                    "primitive_type": "cube",
                    "name": "VectraCube",
                    "location": [0.0, 0.0, 0.0],
                },
            ),
            ActionModel(
                action_id="move_cube",
                tool="object.transform",
                params={
                    "object_name": {"$ref": "create_cube.object_name"},
                    "location": [2.0, 0.0, 0.0],
                },
            ),
        ],
    )
