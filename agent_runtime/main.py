from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .agent.service import AgentService
    from .models import (
        ActionModel,
        AgentStepRequest,
        AgentStepResponse,
        HealthResponse,
        TaskCreateRequest,
        TaskCreateResponse,
    )
    from .planner import plan
    from .utils import model_to_dict
    from vectra.utils.logging import get_vectra_logger, log_structured
except ImportError:  # pragma: no cover - supports `uvicorn main:app` from agent_runtime/
    from agent.service import AgentService
    from models import (
        ActionModel,
        AgentStepRequest,
        AgentStepResponse,
        HealthResponse,
        TaskCreateRequest,
        TaskCreateResponse,
    )
    from planner import plan
    from utils import model_to_dict
    from vectra.utils.logging import get_vectra_logger, log_structured

logger = get_vectra_logger("vectra.runtime")

app = FastAPI(title="Vectra Runtime", version="0.1.0")
agent_service = AgentService()


def _build_action_models(actions: list[dict[str, Any]]) -> list[ActionModel]:
    return [ActionModel.model_validate(action) for action in actions]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/task/create", response_model=TaskCreateResponse)
def create_task(request: TaskCreateRequest) -> TaskCreateResponse:
    request_dict = model_to_dict(request)
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
    log_structured(logger, "task_response", model_to_dict(response))
    return response


@app.post("/agent/step", response_model=AgentStepResponse)
def agent_step(request: AgentStepRequest) -> AgentStepResponse:
    request_dict = model_to_dict(request)
    log_structured(logger, "agent_step_request", request_dict)
    response = agent_service.step(request)
    log_structured(logger, "agent_step_response", model_to_dict(response))
    return response
