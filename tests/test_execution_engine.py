from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from vectra.execution.engine import ExecutionEngine, execute_action
from vectra.tools.base import BaseTool, ToolExecutionError, ToolExecutionResult, ToolValidationError
from vectra.tools.registry import ToolRegistry
from vectra.utils.logging import get_vectra_logger


@dataclass
class FakeContext:
    created_objects: list[str] = field(default_factory=list)
    transforms: list[dict[str, object]] = field(default_factory=list)


class FakeCreatePrimitiveTool(BaseTool):
    name = "mesh.create_primitive"
    description = "Fake create primitive tool"
    input_schema = {
        "primitive_type": {"type": "string", "required": True},
        "name": {"type": "string", "required": False},
        "location": {"type": "vector3", "required": False},
    }

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
    input_schema = {
        "object_name": {"type": "string", "required": True},
        "location": {"type": "vector3", "required": False},
    }

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


class FakeNullableTransformTool(BaseTool):
    name = "object.transform"
    description = "Transform tool that rejects nullable delta values if they survive normalization"
    input_schema = {
        "target": {"type": "string", "required": True},
        "location": {"type": "vector3", "required": False},
        "delta": {"type": "vector3", "required": False},
    }

    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        params = super().validate_params(params)
        if params.get("delta") is None and "delta" in params:
            raise ToolValidationError("'delta' should have been dropped before validation")
        return params

    def execute(self, context: FakeContext, params: dict[str, object]) -> ToolExecutionResult:
        context.transforms.append({"target": params["target"], "location": params.get("location")})
        return ToolExecutionResult(outputs={"object_name": str(params["target"])}, message="normalized")


class FakePlaceOnSurfaceTool(BaseTool):
    name = "object.place_on_surface"
    description = "Fake placement tool"
    input_schema = {
        "target": {"type": "string", "required": True},
        "reference": {"type": "string", "required": True},
        "surface": {"type": "string", "required": False},
    }

    def validate_params(self, params: dict[str, object]) -> dict[str, object]:
        params = super().validate_params(params)
        if not isinstance(params.get("target"), str):
            raise ToolValidationError("'target' must be a string", invalid_params=["target"])
        if not isinstance(params.get("reference"), str):
            raise ToolValidationError("'reference' must be a string", invalid_params=["reference"])
        return params

    def execute(self, context: FakeContext, params: dict[str, object]) -> ToolExecutionResult:
        context.transforms.append({"target": params["target"], "reference": params["reference"]})
        return ToolExecutionResult(outputs={"object_name": str(params["target"])}, message="placed")


class ExecutionErrorTool(BaseTool):
    name = "object.execution_error"
    input_schema = {"target": {"type": "string", "required": True}}

    def execute(self, context: FakeContext, params: dict[str, object]) -> ToolExecutionResult:
        raise ToolExecutionError("Tool exploded safely")


class RuntimeErrorTool(BaseTool):
    name = "object.runtime_error"
    input_schema = {"target": {"type": "string", "required": True}}

    def execute(self, context: FakeContext, params: dict[str, object]) -> ToolExecutionResult:
        raise RuntimeError("Unexpected boom")


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
    assert report.results[0].status == "error"
    assert report.results[0].error_type == "unknown_tool"


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
    assert report.results[0].error_type == "tool_validation_failure"


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
    assert report.results[0].error_type == "reference_resolution_failure"


def test_backend_action_to_execution_loop_works() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool, FakeTransformTool)
    context = FakeContext()
    payload = {"status": "ok", "message": "planned", "actions": EXECUTION_LOOP_ACTIONS}

    report = engine.run(context, payload["actions"])

    assert payload["status"] == "ok"
    assert payload["message"] == "planned"
    assert report.success is True
    assert report.results[-1].outputs == {"object_name": "VectraCube"}


def test_execution_engine_drops_absent_optional_fields_before_validation() -> None:
    engine = _make_engine(FakeNullableTransformTool)
    context = FakeContext()

    report = engine.run(
        context,
        [
            {
                "action_id": "move",
                "tool": "object.transform",
                "params": {"target": "Cube", "location": [0.0, 0.0, 0.0], "delta": None},
            }
        ],
    )

    assert report.success is True
    assert any("Dropped absent optional field 'delta'" in repair for repair in report.repairs)


def test_execute_action_returns_success_shape() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool)
    context = FakeContext()

    result = engine.execute_action(
        context,
        {
            "action_id": "create",
            "tool": "mesh.create_primitive",
            "params": {"primitive_type": "cube", "name": "Cube"},
        },
    )

    assert result.status == "success"
    assert result.tool == "mesh.create_primitive"
    assert result.result == {"object_name": "Cube"}
    assert result.outputs == {"object_name": "Cube"}


def test_module_execute_action_uses_structured_wrapper() -> None:
    registry = ToolRegistry()
    registry.register(FakeCreatePrimitiveTool)

    result = execute_action(
        {
            "action_id": "create",
            "tool": "mesh.create_primitive",
            "params": {"primitive_type": "cube", "name": "Cube"},
        },
        context=FakeContext(),
        registry=registry,
        logger=get_vectra_logger("vectra.tests.execution"),
    )

    assert result.status == "success"
    assert result.result["object_name"] == "Cube"


def test_missing_required_param_returns_structured_validation_failure_without_execution() -> None:
    engine = _make_engine(FakePlaceOnSurfaceTool)
    context = FakeContext()

    report = engine.run(
        context,
        [
            {
                "action_id": "place",
                "tool": "object.place_on_surface",
                "params": {"target": "Lamp"},
            }
        ],
    )

    assert report.success is False
    assert context.transforms == []
    result = report.results[0]
    assert result.status == "error"
    assert result.error_type == "tool_validation_failure"
    assert result.missing_params == ["reference"]
    assert result.invalid_params == []
    assert result.details["required_params"] == ["reference"]


def test_non_mapping_action_returns_structured_action_validation_failure() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool)

    report = engine.run(FakeContext(), ["not an action"])

    assert report.success is False
    assert report.results[0].status == "error"
    assert report.results[0].error_type == "action_validation_failure"
    assert "must be a mapping" in report.results[0].message


def test_non_mapping_params_returns_structured_action_validation_failure() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool)

    report = engine.run(
        FakeContext(),
        [{"action_id": "bad", "tool": "mesh.create_primitive", "params": ["cube"]}],
    )

    assert report.success is False
    assert report.failed_action_id == "bad"
    assert report.results[0].error_type == "action_validation_failure"
    assert "params must be a mapping" in report.results[0].message


def test_non_sequence_actions_returns_structured_report_instead_of_raising() -> None:
    engine = _make_engine(FakeCreatePrimitiveTool)

    report = engine.run(FakeContext(), "not a list")

    assert report.success is False
    assert report.results[0].tool == "<invalid>"
    assert report.results[0].error_type == "action_validation_failure"
    assert report.results[0].details["received_type"] == "str"


def test_tool_execution_error_is_caught_and_structured() -> None:
    engine = _make_engine(ExecutionErrorTool)

    report = engine.run(
        FakeContext(),
        [{"action_id": "explode", "tool": "object.execution_error", "params": {"target": "Cube"}}],
    )

    assert report.success is False
    assert report.results[0].error_type == "tool_execution_failure"
    assert report.results[0].message == "Tool exploded safely"


def test_unexpected_exception_is_caught_and_structured() -> None:
    engine = _make_engine(RuntimeErrorTool)

    report = engine.run(
        FakeContext(),
        [{"action_id": "boom", "tool": "object.runtime_error", "params": {"target": "Cube"}}],
    )

    assert report.success is False
    assert report.results[0].error_type == "unexpected_execution_error"
    assert report.results[0].details["exception_type"] == "RuntimeError"


def test_place_on_surface_auto_fills_exact_floor_reference() -> None:
    engine = _make_engine(FakePlaceOnSurfaceTool)
    context = FakeContext()
    context.scene = SimpleNamespace(objects=[SimpleNamespace(name="floor"), SimpleNamespace(name="Lamp")])

    report = engine.run(
        context,
        [
            {
                "action_id": "place",
                "tool": "object.place_on_surface",
                "params": {"target": "Lamp"},
            }
        ],
    )

    assert report.success is True
    assert context.transforms == [{"target": "Lamp", "reference": "floor"}]
    assert report.results[0].repairs == ["Auto-filled missing reference with exact floor object 'floor'"]
    assert report.repairs == ["Auto-filled missing reference with exact floor object 'floor'"]


def test_place_on_surface_does_not_auto_fill_ambiguous_floor_reference() -> None:
    engine = _make_engine(FakePlaceOnSurfaceTool)
    context = FakeContext()
    context.scene = SimpleNamespace(objects=[SimpleNamespace(name="floor"), SimpleNamespace(name="Floor")])

    report = engine.run(
        context,
        [
            {
                "action_id": "place",
                "tool": "object.place_on_surface",
                "params": {"target": "Lamp"},
            }
        ],
    )

    assert report.success is False
    assert context.transforms == []
    assert report.results[0].error_type == "tool_validation_failure"
    assert report.results[0].missing_params == ["reference"]
