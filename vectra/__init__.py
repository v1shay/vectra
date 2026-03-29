from __future__ import annotations

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
    from .operators.run_task import VECTRA_OT_run_task, cleanup_request_state
    from .tools.registry import get_default_registry
    from .ui.panel import VECTRA_PT_panel

    CLASSES = (
        VECTRA_OT_run_task,
        VECTRA_PT_panel,
    )

    def register() -> None:
        get_default_registry().discover()
        for cls in CLASSES:
            bpy.utils.register_class(cls)

        bpy.types.Scene.vectra_prompt = bpy.props.StringProperty(
            name="Prompt",
            description="Instruction sent to the local Vectra runtime",
            default="",
        )
        bpy.types.Scene.vectra_status = bpy.props.StringProperty(
            name="Status",
            description="Latest Vectra runtime status",
            default="Idle",
        )
        bpy.types.Scene.vectra_phase = bpy.props.StringProperty(
            name="Phase",
            description="Current Vectra request phase",
            default="idle",
        )
        bpy.types.Scene.vectra_request_in_flight = bpy.props.BoolProperty(
            name="Request In Flight",
            description="Whether a Vectra runtime request is currently running",
            default=False,
        )

    def unregister() -> None:
        cleanup_request_state()

        for attr_name in (
            "vectra_request_in_flight",
            "vectra_phase",
            "vectra_status",
            "vectra_prompt",
        ):
            if hasattr(bpy.types.Scene, attr_name):
                delattr(bpy.types.Scene, attr_name)

        for cls in reversed(CLASSES):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass

else:
    CLASSES = ()

    def register() -> None:
        raise RuntimeError("The Vectra Blender add-on can only be registered inside Blender")

    def unregister() -> None:
        return None


if __name__ == "__main__" and bpy is not None:
    register()
