from __future__ import annotations

import pytest

from vectra.tools.base import BaseTool, ToolExecutionResult
from vectra.tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry


class EchoTool(BaseTool):
    name = "test.echo"
    description = "Echo tool"
    input_schema = {"value": {"type": "string"}}

    def execute(self, context, params):
        del context
        return ToolExecutionResult(outputs={"value": params["value"]}, message="echoed")


class DuplicateEchoTool(BaseTool):
    name = "test.echo"
    description = "Duplicate echo tool"
    input_schema = {}

    def execute(self, context, params):
        del context
        del params
        return ToolExecutionResult()


def test_tool_registry_register_and_lookup_work() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool)

    tool = registry.get("test.echo")

    assert tool.name == "test.echo"
    assert registry.list_tools() == ["test.echo"]


def test_tool_registry_duplicate_name_fails_clearly() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool)

    with pytest.raises(DuplicateToolError):
        registry.register(DuplicateEchoTool)


def test_tool_registry_discover_loads_decorated_tools() -> None:
    registry = ToolRegistry()
    registry.discover()

    discovered_tools = registry.list_tools()

    assert "mesh.create_primitive" in discovered_tools
    assert "object.transform" in discovered_tools
    assert registry.get("mesh.create_primitive").output_schema == {
        "object_name": {"type": "string"}
    }


def test_tool_registry_unknown_tool_lookup_raises_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("missing.tool")
