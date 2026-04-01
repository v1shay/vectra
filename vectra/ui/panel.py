from __future__ import annotations

import bpy

from .. import addon_loader
from ..addon_bootstrap import current_dev_source_path
from ..operators.run_task import get_reload_block_reason


class VECTRA_PT_panel(bpy.types.Panel):
    bl_label = "Vectra"
    bl_idname = "VECTRA_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Vectra"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "vectra_prompt", text="Prompt")

        run_col = layout.column()
        run_col.enabled = not scene.vectra_request_in_flight
        run_col.operator("vectra.run_task", text="Run Task")

        layout.label(text=f"Status: {scene.vectra_status}")
        layout.label(text=f"Phase: {scene.vectra_phase}")

        status = addon_loader.get_runtime_status()
        configured_source = current_dev_source_path(context)
        dev_box = layout.box()
        dev_box.label(text=f"Runtime: {status.mode}")
        dev_box.label(
            text=f"Source: {status.source_path or configured_source or 'Packaged add-on'}"
        )
        if status.mode == "packaged" and not configured_source:
            dev_box.label(text="Set Dev Source Path to the repo root for backend auto-start.")
            dev_box.label(text="Packaged mode without a repo path can only talk to an already-running backend.")
        block_reason = get_reload_block_reason()
        if block_reason:
            dev_box.label(text=f"Reload blocked: {block_reason}")
        if status.error:
            dev_box.label(text=f"Last reload error: {status.error}")
        dev_box.operator("vectra.reload_dev", text="Reload Vectra Dev")
