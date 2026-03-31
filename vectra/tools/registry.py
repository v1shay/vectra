from __future__ import annotations

import importlib
import pkgutil
import sys
from typing import Any

from .base import BaseTool

_TOOL_CLASS_CATALOG: dict[str, type[BaseTool]] = {}


class ToolRegistryError(Exception):
    """Base exception for registry errors."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool name is registered more than once."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a tool lookup fails."""


def _tool_name_from_class(tool_cls: type[BaseTool]) -> str:
    tool_name = getattr(tool_cls, "name", "")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ToolRegistryError(f"Tool class '{tool_cls.__name__}' must define a non-empty name")
    return tool_name


def register_tool(tool_cls: type[BaseTool]) -> type[BaseTool]:
    tool_name = _tool_name_from_class(tool_cls)
    existing = _TOOL_CLASS_CATALOG.get(tool_name)
    if existing is not None and existing is not tool_cls:
        if existing.__module__ == tool_cls.__module__ and existing.__name__ == tool_cls.__name__:
            _TOOL_CLASS_CATALOG[tool_name] = tool_cls
            return tool_cls
        raise DuplicateToolError(f"Tool '{tool_name}' is already registered")
    _TOOL_CLASS_CATALOG[tool_name] = tool_cls
    return tool_cls


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._discovered_modules: set[str] = set()

    def register(self, tool_cls: type[BaseTool]) -> type[BaseTool]:
        tool_name = _tool_name_from_class(tool_cls)
        if tool_name in self._tools:
            raise DuplicateToolError(f"Tool '{tool_name}' is already registered in this registry")
        self._tools[tool_name] = tool_cls()
        return tool_cls

    def get(self, tool_name: str) -> BaseTool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Unknown tool '{tool_name}'") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def reset(self) -> None:
        self._tools.clear()
        self._discovered_modules.clear()

    def discover(self) -> None:
        package_name = __name__.rsplit(".", 1)[0]
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            short_name = module_info.name.rsplit(".", 1)[-1]
            if short_name in {"base", "registry", "__init__"}:
                continue
            if module_info.name in self._discovered_modules:
                continue
            module = sys.modules.get(module_info.name)
            if module is None:
                importlib.import_module(module_info.name)
            else:
                importlib.reload(module)
            self._discovered_modules.add(module_info.name)

        for tool_name, tool_cls in _TOOL_CLASS_CATALOG.items():
            if tool_name not in self._tools:
                self.register(tool_cls)


_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    global _DEFAULT_REGISTRY

    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry()
    return _DEFAULT_REGISTRY


def reset_default_registry() -> ToolRegistry:
    global _DEFAULT_REGISTRY

    _TOOL_CLASS_CATALOG.clear()
    _DEFAULT_REGISTRY = ToolRegistry()
    return _DEFAULT_REGISTRY
