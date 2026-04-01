from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None


if bpy is not None:
    from .operators.run_task import (
        VECTRA_OT_run_task,
        cleanup_request_state,
        get_reload_block_reason as _get_reload_block_reason,
    )
    from .tools.registry import get_default_registry, reset_default_registry
    from .ui.panel import VECTRA_PT_panel

    CLASSES = (
        VECTRA_OT_run_task,
        VECTRA_PT_panel,
    )
    _SCENE_PROPERTIES = {
        "vectra_prompt": lambda: bpy.props.StringProperty(
            name="Prompt",
            description="Instruction sent to the local Vectra runtime",
            default="",
        ),
        "vectra_status": lambda: bpy.props.StringProperty(
            name="Status",
            description="Latest Vectra runtime status",
            default="Idle",
        ),
        "vectra_phase": lambda: bpy.props.StringProperty(
            name="Phase",
            description="Current Vectra request phase",
            default="idle",
        ),
        "vectra_request_in_flight": lambda: bpy.props.BoolProperty(
            name="Request In Flight",
            description="Whether a Vectra runtime request is currently running",
            default=False,
        ),
        "vectra_execution_mode": lambda: bpy.props.EnumProperty(
            name="Execution Mode",
            description="Execution backend used when VECTRA_AGENT_MODE is enabled",
            items=(
                ("vectra-dev", "vectra-dev", "Tool-based execution"),
                ("vectra-code", "vectra-code", "Console-injected Python execution"),
            ),
            default="vectra-dev",
        ),
        "vectra_agent_transcript": lambda: bpy.props.StringProperty(
            name="Agent Transcript",
            description="Narrated progress from the autonomous agent loop",
            default="",
        ),
        "vectra_pending_question": lambda: bpy.props.StringProperty(
            name="Pending Question",
            description="Clarification requested by the agent loop",
            default="",
        ),
        "vectra_iteration": lambda: bpy.props.IntProperty(
            name="Agent Iteration",
            description="Current iteration of the agent loop",
            default=0,
            min=0,
        ),
    }

    def _is_already_registered_error(exc: Exception) -> bool:
        return "already registered" in str(exc).lower()


    def _registered_bpy_type(cls: type[object]) -> object | None:
        bpy_types = getattr(bpy, "types", None)
        if bpy_types is None:
            return None

        candidate_ids = [cls.__name__]
        bl_idname = getattr(cls, "bl_idname", None)
        if isinstance(bl_idname, str) and bl_idname and bl_idname not in candidate_ids:
            candidate_ids.append(bl_idname)

        base_types = []
        for base_type_name in ("AddonPreferences", "Operator", "Panel"):
            base_type = getattr(bpy_types, base_type_name, None)
            if base_type is not None:
                base_types.append(base_type)

        for base_type in base_types:
            finder = getattr(base_type, "bl_rna_get_subclass_py", None)
            if callable(finder):
                for candidate_id in candidate_ids:
                    existing = finder(candidate_id, None)
                    if existing is not None:
                        return existing

        for candidate_id in candidate_ids:
            existing = getattr(bpy_types, candidate_id, None)
            if existing is not None:
                return existing
        return None


    def _register_class_safe(cls: type[object]) -> None:
        existing = _registered_bpy_type(cls)
        if existing is cls:
            return
        if existing is not None and existing is not cls:
            try:
                bpy.utils.unregister_class(existing)
            except Exception:
                pass

        try:
            bpy.utils.register_class(cls)
        except Exception as exc:
            if not _is_already_registered_error(exc):
                raise
            existing = _registered_bpy_type(cls)
            if existing is cls:
                return
            if existing is not None and existing is not cls:
                try:
                    bpy.utils.unregister_class(existing)
                except Exception:
                    pass
                bpy.utils.register_class(cls)


    def _unregister_class_safe(cls: type[object]) -> None:
        existing = _registered_bpy_type(cls)
        target = existing if existing is not None else cls
        try:
            bpy.utils.unregister_class(target)
        except Exception:
            pass


    def _register_scene_properties() -> None:
        for property_name, factory in _SCENE_PROPERTIES.items():
            if not hasattr(bpy.types.Scene, property_name):
                setattr(bpy.types.Scene, property_name, factory())


    def _unregister_scene_properties() -> None:
        for property_name in _SCENE_PROPERTIES:
            if hasattr(bpy.types.Scene, property_name):
                delattr(bpy.types.Scene, property_name)


    def get_reload_block_reason() -> str | None:
        return _get_reload_block_reason()


    def register() -> None:
        cleanup_request_state()
        reset_default_registry()
        get_default_registry().discover()
        for cls in CLASSES:
            _register_class_safe(cls)
        _register_scene_properties()


    def unregister() -> None:
        cleanup_request_state()
        _unregister_scene_properties()
        for cls in reversed(CLASSES):
            _unregister_class_safe(cls)
        reset_default_registry()


else:
    CLASSES = ()

    def get_reload_block_reason() -> str | None:
        return None


    def register() -> None:
        raise RuntimeError("The Vectra Blender runtime can only be registered inside Blender")


    def unregister() -> None:
        return None
