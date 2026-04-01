from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

_RESULT_KEY = "vectra_last_console_result"
_BANNED_PATTERNS = (
    "import os",
    "import subprocess",
    "import socket",
    "from os",
    "from subprocess",
    "open(",
    "pathlib.",
    "bpy.ops.wm.quit_blender",
    "bpy.ops.preferences.",
    "addon_enable",
    "addon_disable",
    "addon_install",
)


class CodeExecutionError(Exception):
    """Raised when the vectra-code executor cannot run a snippet safely."""


@dataclass
class CodeExecutionReport:
    success: bool
    message: str
    result: dict[str, Any] = field(default_factory=dict)


class ConsoleCodeExecutor:
    def validate(self, code: str) -> str:
        normalized = code.strip()
        if not normalized:
            raise CodeExecutionError("vectra-code produced an empty code snippet")

        lowered = normalized.lower()
        for pattern in _BANNED_PATTERNS:
            if pattern in lowered:
                raise CodeExecutionError(f"vectra-code blocked unsafe pattern: {pattern}")
        return normalized

    def _console_override(self) -> tuple[dict[str, Any], Any, str]:
        if bpy is None:
            raise CodeExecutionError("Blender Python API is unavailable")

        windows = getattr(getattr(bpy.context, "window_manager", None), "windows", None)
        if not windows:
            raise CodeExecutionError("No Blender window is available for console execution")

        window = windows[0]
        screen = getattr(window, "screen", None)
        if screen is None or not getattr(screen, "areas", None):
            raise CodeExecutionError("No Blender screen areas are available for console execution")

        for area in screen.areas:
            if area.type in {"CONSOLE", "VIEW_3D", "TEXT_EDITOR"}:
                original_type = area.type
                area.type = "CONSOLE"
                region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None)
                if region is None:
                    area.type = original_type
                    continue
                return (
                    {
                        "window": window,
                        "screen": screen,
                        "area": area,
                        "region": region,
                    },
                    area,
                    original_type,
                )
        raise CodeExecutionError("No Blender area could be converted into a console for vectra-code")

    def _wrap_code(self, code: str) -> str:
        source = (
            "import bpy, traceback\n"
            f"_vectra_code = {json.dumps(code)}\n"
            "namespace = {'bpy': bpy}\n"
            "try:\n"
            "    exec(compile(_vectra_code, '<vectra-code>', 'exec'), namespace, namespace)\n"
            f"    bpy.app.driver_namespace[{json.dumps(_RESULT_KEY)}] = {{\n"
            "        'success': True,\n"
            "        'message': 'Executed vectra-code snippet successfully',\n"
            "        'namespace_keys': sorted(key for key in namespace if not key.startswith('__')),\n"
            "    }\n"
            "except Exception as exc:\n"
            f"    bpy.app.driver_namespace[{json.dumps(_RESULT_KEY)}] = {{\n"
            "        'success': False,\n"
            "        'message': str(exc),\n"
            "        'traceback': traceback.format_exc(),\n"
            "    }\n"
        )
        return f"exec(compile({json.dumps(source)}, '<vectra-console-wrapper>', 'exec'), globals(), globals())"

    def run(self, context: Any, code: str) -> CodeExecutionReport:
        del context
        normalized = self.validate(code)
        override, area, original_type = self._console_override()
        driver_namespace = getattr(bpy.app, "driver_namespace", None)
        if not isinstance(driver_namespace, dict):
            raise CodeExecutionError("Blender driver namespace is unavailable for vectra-code results")

        wrapped = self._wrap_code(normalized)
        driver_namespace.pop(_RESULT_KEY, None)
        try:
            area.spaces.active.language = "python"
            with bpy.context.temp_override(**override):
                bpy.ops.console.language(language="python")
                bpy.ops.console.insert(text=wrapped)
                bpy.ops.console.execute(interactive=False)
        except RuntimeError as exc:
            raise CodeExecutionError(f"vectra-code console execution failed: {exc}") from exc
        finally:
            area.type = original_type

        result = driver_namespace.get(_RESULT_KEY)
        if not isinstance(result, dict):
            raise CodeExecutionError("vectra-code did not publish an execution result")
        success = bool(result.get("success"))
        message = str(result.get("message", "vectra-code execution completed")).strip()
        return CodeExecutionReport(success=success, message=message, result=dict(result))
