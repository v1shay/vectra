from __future__ import annotations

import logging

from fastapi import FastAPI

from .models import HealthResponse, TaskCreateRequest, TaskCreateResponse
from .planner_stub import build_task_response

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
    return build_task_response(
        prompt=request.prompt,
        scene_state=request.scene_state,
        images=request.images,
    )
