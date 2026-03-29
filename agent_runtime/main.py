from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .models import HealthResponse, TaskCreateRequest, TaskCreateResponse
    from .planner import plan
except ImportError:  # pragma: no cover - supports `uvicorn main:app` from agent_runtime/
    from models import HealthResponse, TaskCreateRequest, TaskCreateResponse
    from planner import plan

logger = logging.getLogger("vectra.runtime")

app = FastAPI(title="Vectra Runtime", version="0.1.0")


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[attr-defined,no-any-return]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/task/create", response_model=TaskCreateResponse)
def create_task(request: TaskCreateRequest) -> TaskCreateResponse:
    logger.info("Received task request: %s", _model_to_dict(request))
    planner_result = plan(
        request.prompt,
        request.scene_state,
    )
    return TaskCreateResponse(
        status="ok",
        message=planner_result.message,
        actions=planner_result.actions,
    )
