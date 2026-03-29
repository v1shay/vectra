from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from agent_runtime.main import app
from vectra.execution.engine import ExecutionEngine
from vectra.tools.base import BaseTool, ToolExecutionResult, ToolValidationError
from vectra.tools.registry import ToolRegistry
from vectra.utils.logging import get_vectra_logger


@dataclass
class KernelContext:
    created_objects: list[str] = field(default_factory=list)
    transforms: list[dict[str, object]] = field(default_factory=list)


class KernelCreatePrimitiveTool(BaseTool):
    name = "mesh.create_primitive"
    description = "Kernel integration test create tool"
    input_schema = {}

    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        params = super().validate_params(params)
        if params.get("primitive_type") != "cube":
            raise ToolValidationError("Expected cube primitive")
        if not isinstance(params.get("name"), str):
            raise ToolValidationError("'name' is required")
        return params

    def execute(self, context: KernelContext, params: dict[str, object]) -> ToolExecutionResult:
        context.created_objects.append(params["name"])
        return ToolExecutionResult(outputs={"object_name": params["name"]}, message="created")


class KernelTransformTool(BaseTool):
    name = "object.transform"
    description = "Kernel integration test transform tool"
    input_schema = {}

    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        params = super().validate_params(params)
        if not isinstance(params.get("object_name"), str):
            raise ToolValidationError("'object_name' must be a string")
        if not isinstance(params.get("location"), list):
            raise ToolValidationError("'location' must be a list")
        return params

    def execute(self, context: KernelContext, params: dict[str, object]) -> ToolExecutionResult:
        context.transforms.append(
            {"object_name": params["object_name"], "location": params["location"]}
        )
        return ToolExecutionResult(outputs={"object_name": params["object_name"]}, message="moved")


class FailingTransformTool(KernelTransformTool):
    def execute(self, context: KernelContext, params: dict[str, object]) -> ToolExecutionResult:
        del context
        del params
        raise ToolValidationError("Transform execution failed")


def _make_engine(*tool_classes: type[BaseTool]) -> ExecutionEngine:
    registry = ToolRegistry()
    for tool_cls in tool_classes:
        registry.register(tool_cls)
    return ExecutionEngine(registry=registry, logger=get_vectra_logger("vectra.tests.kernel"))


def _create_payload() -> dict[str, object]:
    client = TestClient(app)
    response = client.post(
        "/task/create",
        json={
            "prompt": "kernel integration test",
            "scene_state": {
                "active_object": None,
                "selected_objects": [],
                "current_frame": 1,
            },
            "images": [],
        },
    )
    assert response.status_code == 200
    return response.json()


def test_kernel_integration_create_then_transform_sequence_executes() -> None:
    engine = _make_engine(KernelCreatePrimitiveTool, KernelTransformTool)
    context = KernelContext()
    payload = _create_payload()

    report = engine.run(context, payload["actions"])

    assert payload["status"] == "ok"
    assert payload["message"] == "planned"
    assert report.success is True
    assert report.message == "Executed 2 action(s) successfully"
    assert context.created_objects == ["VectraCube"]
    assert context.transforms == [
        {"object_name": "VectraCube", "location": [2.0, 0.0, 0.0]}
    ]


def test_kernel_integration_failure_returns_structured_error() -> None:
    engine = _make_engine(KernelCreatePrimitiveTool, FailingTransformTool)
    context = KernelContext()
    payload = _create_payload()

    report = engine.run(context, payload["actions"])

    assert report.success is False
    assert report.failed_action_id == "move_cube"
    assert report.results[-1].error == "Transform execution failed"
