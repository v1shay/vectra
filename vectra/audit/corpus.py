from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditCase:
    prompt: str
    setup_id: str
    expects_scene: bool = False
    expects_composite: bool = False
    expects_animation: bool = False
    expects_lighting: bool = False
    expects_camera: bool = False
    modes: tuple[str, ...] | None = None


DEFAULT_MODES = ("vectra-dev", "vectra-code")
DEFAULT_SETUP_IDS = (
    "empty-scene",
    "plain-geometric-scene",
    "cluttered-irrelevant-object-scene",
    "half-built-scene",
    "intentionally-awkward-layout-scene",
    "simple-lit-scene-for-animation",
    "transient-execution-failure-scene",
)

AUDIT_CASES: tuple[AuditCase, ...] = (
    AuditCase(
        (
            "Create a coherent, cinematic interior scene with one clear focal point. Build real structure, "
            "not random primitives: include a floor, surrounding room elements, and at least one intentional "
            "multi-part object that reads as a designed piece of furniture or decor. Arrange the scene so the "
            "focal point is obvious, reduce overlap, add lighting that makes the scene readable and appealing, "
            "and frame it with the camera. Then add a short visible animation where either the camera or one of "
            "the lights moves across the scene in a way that improves the presentation. Finish only when the "
            "result feels intentionally composed rather than just populated with objects."
        ),
        "empty-scene",
        expects_scene=True,
        expects_composite=True,
        expects_animation=True,
        expects_lighting=True,
        expects_camera=True,
        modes=("vectra-dev",),
    ),
    AuditCase("make something cool", "empty-scene", expects_scene=True, expects_lighting=True, expects_camera=True),
    AuditCase(
        "build a futuristic room with floating objects",
        "empty-scene",
        expects_scene=True,
        expects_composite=True,
        expects_lighting=True,
        expects_camera=True,
    ),
    AuditCase("make a desk with a lamp and chair", "empty-scene", expects_composite=True),
    AuditCase("fix the spacing and improve the composition", "intentionally-awkward-layout-scene", expects_scene=True),
    AuditCase("add lighting and make it look cinematic", "plain-geometric-scene", expects_scene=True, expects_lighting=True, expects_camera=True),
    AuditCase("animate the camera around the scene", "simple-lit-scene-for-animation", expects_animation=True, expects_camera=True),
    AuditCase("animate a light sweeping across the objects", "simple-lit-scene-for-animation", expects_animation=True, expects_lighting=True),
    AuditCase("clean this scene up and make it coherent", "cluttered-irrelevant-object-scene", expects_scene=True),
    AuditCase("finish this and make it coherent", "half-built-scene", expects_scene=True, expects_composite=True),
)
