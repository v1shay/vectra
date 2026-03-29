from __future__ import annotations

from dataclasses import dataclass, field

from vectra.execution.engine import ExecutionEngine
from vectra.tools.base import BaseTool, ToolExecutionResult, ToolValidationError
from vectra.tools.registry import ToolRegistry
from vectra.utils.logging import get_vectra_logger


@dataclass
class FakeContext:
    created_objects: list[str] = field(default_factory=list)
    transforms: list[dict[str, object]] = field(default_factory=list)


class FakeCreatePrimitiveTool(BaseTool):
    name = "mesh.create_primitive"
    description = "Fake create primitive tool"
    input_schema = {}

    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        params = super().validate_params(params)
        if params.get("primitive_type") != "cube":
            raise ToolValidationError("Only cube primitives are supported in this test")
        if not isinstance(params.get("name"), str):
            raise ToolValidationError("'name' is required")
        return params

    def execute(self, context: FakeContext, params: dict[str, object]) -> ToolExecutionResult:
        context.created_objects.append(params["name"])
        return ToolExecutionResult(
            outputs={"object_name": params["name"]},
            message="created",
        )


class FakeTransformTool(BaseTool):
    name = "object.transform"
    description = "Fake transform tool"
    input_schema = {}

    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        params = super().validate_params(params)
        if not isinstance(params.get("object_name"), str):
            raise ToolValidationError("'object_name' must be resolved to a string")
        if not isinstance(params.get("location"), list):
            raise ToolValidationError("'location' must be a list")
        return params

    def execute(self, context: FakeContext, params: dict[str, object]) -> ToolExecutionResult:
        context.transforms.append(
            {
                "object_name": params["object_name"],
                "location": params["location"],
            }
        )
        return ToolExecutionResult(outputs={"object_name": params["object_name"]}, message="moved")


class RejectingTransformTool(FakeTransformTool):
    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        raise ToolValidationError("Invalid transform params")


EXECUTION_LOOP_ACTIONS = [
    {
        "action_id": "create_cube",
        "tool": "mesh.create_primitive",
        "params": {
            "primitive_type": "cube",
            "name": "VectraCube",
            "location": [0.0, 0.0, 0.0],
        },
    },
    {
        "action_id": "move_cube",
        "tool": "object.transform",
        "params": {
            "object_name": {"$ref": "create_cube.object_name"},
            "location": [2.0, 0.0, 0.0],
        },
    },
]


def _make_engine(*tool_classes: type[BaseTool]) -> ExecutionEngine:
    registry = ToolRegistry()
    for tool_cls in tool_classes:
        registry.register(tool_cls)
    return ExecutionEngine(registry=registry, logger=get_vectra_logger("vectra.tests.execution"))


def test_execution_engine_runs_actions_sequentially_with_refs() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool, FakeTransformTool)
    context = FakeContext()
    report = engine.run(context, EXECUTION_LOOP_ACTIONS)

    assert report.success is True
    assert report.message == "Executed 2 action(s) successfully"
    assert context.created_objects == ["VectraCube"]
    assert context.transforms == [
        {"object_name": "VectraCube", "location": [2.0, 0.0, 0.0]}
    ]


def test_execution_engine_invalid_tool_fails_safely() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool)
    context = FakeContext()

    report = engine.run(
        context,
        [{"action_id": "missing", "tool": "missing.tool", "params": {}}],
    )

    assert report.success is False
    assert report.failed_action_id == "missing"
    assert "missing.tool" in report.message


def test_execution_engine_invalid_params_fail_safely() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool, RejectingTransformTool)
    context = FakeContext()

    report = engine.run(
        context,
        [
            {
                "action_id": "move",
                "tool": "object.transform",
                "params": {"object_name": "Cube", "location": [0, 0, 0]},
            }
        ],
    )

    assert report.success is False
    assert report.failed_action_id == "move"
    assert report.results[0].error == "Invalid transform params"


def test_execution_engine_unresolved_ref_fails_safely() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool, FakeTransformTool)
    context = FakeContext()

    report = engine.run(
        context,
        [
            {
                "action_id": "move",
                "tool": "object.transform",
                "params": {
                    "object_name": {"$ref": "missing.object_name"},
                    "location": [1.0, 2.0, 3.0],
                },
            }
        ],
    )

    assert report.success is False
    assert report.failed_action_id == "move"
    assert "Unknown action reference 'missing'" in report.message


def test_backend_action_to_execution_loop_works() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool, FakeTransformTool)
    context = FakeContext()
    payload = {"status": "ok", "message": "planned", "actions": EXECUTION_LOOP_ACTIONS}

    report = engine.run(context, payload["actions"])

    assert payload["status"] == "ok"
    assert payload["message"] == "planned"
    assert report.success is True
    assert report.results[-1].outputs == {"object_name": "VectraCube"}
