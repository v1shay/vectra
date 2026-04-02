from __future__ import annotations

import pytest

from vectra.tools.base import BaseTool, ToolExecutionResult
from vectra.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)


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


def test_tool_registry_discover_loads_director_tool_surface() -> None:
    registry = ToolRegistry()
    registry.discover()

    discovered_tools = registry.list_tools()

    assert "mesh.create_primitive" in discovered_tools
    assert "object.transform" in discovered_tools
    assert "object.duplicate" in discovered_tools
    assert "scene.group" in discovered_tools
    assert "light.create" in discovered_tools
    assert "camera.ensure" in discovered_tools
    assert "material.apply_basic" in discovered_tools
    assert "scene.get_state" in discovered_tools
    assert "scene.capture_view" in discovered_tools


def test_tool_registry_unknown_tool_lookup_raises_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("missing.tool")


def test_reset_default_registry_clears_catalog_and_discovers_fresh_tools() -> None:
    reset_default_registry()
    registry = get_default_registry()

    assert registry.list_tools() == []

    registry.discover()

    assert "mesh.create_primitive" in registry.list_tools()
    assert "scene.capture_view" in registry.list_tools()
