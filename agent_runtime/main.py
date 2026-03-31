from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .models import ActionModel, HealthResponse, TaskCreateRequest, TaskCreateResponse
    from .planner import plan
    from vectra.utils.logging import get_vectra_logger, log_structured
except ImportError:  # pragma: no cover - supports `uvicorn main:app` from agent_runtime/
    from models import ActionModel, HealthResponse, TaskCreateRequest, TaskCreateResponse
    from planner import plan
    from vectra.utils.logging import get_vectra_logger, log_structured

logger = get_vectra_logger("vectra.runtime")

app = FastAPI(title="Vectra Runtime", version="0.1.0")


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[attr-defined,no-any-return]


def _build_action_models(actions: list[dict[str, Any]]) -> list[ActionModel]:
    return [ActionModel.model_validate(action) for action in actions]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/task/create", response_model=TaskCreateResponse)
def create_task(request: TaskCreateRequest) -> TaskCreateResponse:
    request_dict = _model_to_dict(request)
    log_structured(logger, "task_request", request_dict)
    planner_result = plan(
        request.prompt,
        request.scene_state,
    )
    response = TaskCreateResponse(
        status=planner_result.status,
        message=planner_result.message,
        actions=_build_action_models(planner_result.actions),
    )
    log_structured(logger, "task_response", _model_to_dict(response))
    return response
