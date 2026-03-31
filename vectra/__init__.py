from __future__ import annotations

import sys

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly by tests
    bpy = None

bl_info = {
    "name": "Vectra",
    "author": "OpenAI Codex",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Vectra",
    "description": "Vectra control-loop validation add-on",
    "category": "3D View",
}

if bpy is not None:
    from . import addon_loader
    from .addon_bootstrap import current_dev_source_path, register_bootstrap_classes, unregister_bootstrap_classes

    def register() -> None:
        try:
            addon_loader.deactivate_runtime(sys.modules[__name__], enforce_idle=False)
        except addon_loader.RuntimeLoadError:
            addon_loader.reset_loader_state()
        unregister_bootstrap_classes()
        register_bootstrap_classes()
        addon_loader.activate_runtime(
            sys.modules[__name__],
            dev_source_path=current_dev_source_path(),
        )

    def unregister() -> None:
        try:
            addon_loader.deactivate_runtime(sys.modules[__name__], enforce_idle=False)
        finally:
            unregister_bootstrap_classes()

else:
    def register() -> None:
        raise RuntimeError("The Vectra Blender add-on can only be registered inside Blender")

    def unregister() -> None:
        return None


if __name__ == "__main__" and bpy is not None:
    register()
