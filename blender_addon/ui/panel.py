from __future__ import annotations

import bpy


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
