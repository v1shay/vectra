from __future__ import annotations

import sys
from types import ModuleType

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from . import addon_loader

ADDON_PACKAGE_NAME = __package__ or __name__.rsplit(".", 1)[0]


if bpy is not None:
    def _normalize_path_value(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None

        normalized = value.strip()
        return normalized or None


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


    def _addon_package_candidates() -> list[str]:
        candidates = [ADDON_PACKAGE_NAME]
        short_name = ADDON_PACKAGE_NAME.rsplit(".", 1)[-1]
        if short_name not in candidates:
            candidates.append(short_name)
        return candidates


    def _addon_preferences(context: bpy.types.Context | None = None) -> bpy.types.AddonPreferences | None:
        bpy_context = context or getattr(bpy, "context", None)
        preferences = getattr(bpy_context, "preferences", None)
        addons = getattr(preferences, "addons", None)
        if addons is None:
            return None

        for candidate in _addon_package_candidates():
            addon_entry = addons.get(candidate)
            if addon_entry is not None:
                return getattr(addon_entry, "preferences", None)
        return None


    def _addon_package_module() -> ModuleType:
        package_module = sys.modules.get(ADDON_PACKAGE_NAME)
        if isinstance(package_module, ModuleType):
            return package_module
        raise addon_loader.RuntimeLoadError(
            f"Unable to locate the active Vectra add-on package '{ADDON_PACKAGE_NAME}'"
        )


    def current_dev_source_path(context: bpy.types.Context | None = None) -> str | None:
        preferences = _addon_preferences(context)
        if preferences is None:
            return None
        return _normalize_path_value(getattr(preferences, "dev_source_path", None))


    class VectraAddonPreferences(bpy.types.AddonPreferences):
        bl_idname = ADDON_PACKAGE_NAME

        dev_source_path: bpy.props.StringProperty(
            name="Development Source Path",
            description="Path to the local Vectra repo root for live development reloads",
            subtype="DIR_PATH",
            default="",
        )

        def draw(self, context: bpy.types.Context) -> None:
            del context
            layout = self.layout
            status = addon_loader.get_runtime_status()

            layout.prop(self, "dev_source_path")
            layout.label(text=f"Mode: {status.mode}")
            layout.label(text=f"Resolved source: {status.source_path or 'Packaged add-on'}")
            if status.error:
                layout.label(text=f"Last reload error: {status.error}")
            layout.operator("vectra.reload_dev", text="Reload Development Source")


    class VECTRA_OT_reload_dev(bpy.types.Operator):
        bl_idname = "vectra.reload_dev"
        bl_label = "Reload Development Source"
        bl_description = "Reload the Vectra runtime from the configured development source path"

        def execute(self, context: bpy.types.Context) -> set[str]:
            scene = getattr(context, "scene", None)
            try:
                addon_loader.activate_runtime(
                    _addon_package_module(),
                    dev_source_path=current_dev_source_path(context),
                )
            except addon_loader.RuntimeReloadBlockedError as exc:
                if scene is not None and hasattr(scene, "vectra_status"):
                    scene.vectra_status = str(exc)
                    if hasattr(scene, "vectra_phase"):
                        scene.vectra_phase = "error"
                self.report({"WARNING"}, str(exc))
                return {"CANCELLED"}
            except addon_loader.RuntimeLoadError as exc:
                if scene is not None and hasattr(scene, "vectra_status"):
                    scene.vectra_status = str(exc)
                    if hasattr(scene, "vectra_phase"):
                        scene.vectra_phase = "error"
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            status = addon_loader.get_runtime_status()
            message = (
                f"Reloaded Vectra from {status.source_path}"
                if status.source_path
                else "Reloaded packaged Vectra runtime"
            )
            if scene is not None and hasattr(scene, "vectra_status"):
                scene.vectra_status = message
                if hasattr(scene, "vectra_phase"):
                    scene.vectra_phase = "success"
            self.report({"INFO"}, message)
            return {"FINISHED"}


    BOOTSTRAP_CLASSES = (
        VectraAddonPreferences,
        VECTRA_OT_reload_dev,
    )

    def register_bootstrap_classes() -> None:
        for cls in BOOTSTRAP_CLASSES:
            _register_class_safe(cls)


    def unregister_bootstrap_classes() -> None:
        for cls in reversed(BOOTSTRAP_CLASSES):
            _unregister_class_safe(cls)


else:
    def current_dev_source_path(context=None) -> str | None:
        del context
        return None


    def register_bootstrap_classes() -> None:
        raise RuntimeError("The Vectra Blender bootstrap can only be registered inside Blender")


    def unregister_bootstrap_classes() -> None:
        return None
