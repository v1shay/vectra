from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

_SCREENSHOT_DIR = Path(tempfile.gettempdir()) / "vectra-agent-screenshots"


def _vector_to_list(vector: Any, size: int = 3) -> list[float]:
    return [float(component) for component in vector[:size]]


def _approx_bounds(obj: Any) -> dict[str, list[float]]:
    location = _vector_to_list(obj.location)
    dimensions = _vector_to_list(getattr(obj, "dimensions", (2.0, 2.0, 2.0)))
    half_extents = [dimension / 2.0 for dimension in dimensions]
    return {
        "min": [location[index] - half_extents[index] for index in range(3)],
        "max": [location[index] + half_extents[index] for index in range(3)],
    }


def build_scene_state(context: bpy.types.Context) -> dict[str, Any]:
    if bpy is None:
        raise RuntimeError("Blender Python API is unavailable")
    scene = context.scene
    active_object = context.active_object
    objects = []
    scene_objects = getattr(scene, "objects", [])
    for obj in scene_objects:
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "selected": bool(obj.select_get()),
                "active": active_object is not None and obj == active_object,
                "location": _vector_to_list(obj.location),
                "rotation_euler": _vector_to_list(obj.rotation_euler),
                "scale": _vector_to_list(obj.scale),
                "dimensions": _vector_to_list(getattr(obj, "dimensions", (2.0, 2.0, 2.0))),
                "bounds": _approx_bounds(obj),
                "parent": getattr(getattr(obj, "parent", None), "name", None),
                "children": [child.name for child in getattr(obj, "children", [])],
                "collection_names": [
                    collection.name
                    for collection in getattr(obj, "users_collection", [])
                    if hasattr(collection, "name")
                ],
            }
        )

    return {
        "active_object": active_object.name if active_object else None,
        "selected_objects": [obj.name for obj in context.selected_objects],
        "current_frame": scene.frame_current,
        "objects": objects,
    }


def _view3d_override() -> dict[str, Any] | None:
    if bpy is None:
        return None
    window_manager = getattr(bpy.context, "window_manager", None)
    windows = getattr(window_manager, "windows", None)
    if windows is None:
        return None

    for window in windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in getattr(screen, "areas", []):
            if area.type != "VIEW_3D":
                continue
            region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None)
            if region is None:
                continue
            return {
                "window": window,
                "screen": screen,
                "area": area,
                "region": region,
            }
    return None


def capture_viewport_screenshot(iteration: int) -> dict[str, Any]:
    if bpy is None:
        return {"available": False, "path": None, "reason": "Blender Python API is unavailable"}
    if bool(getattr(getattr(bpy, "app", None), "background", False)):
        return {"available": False, "path": None, "reason": "Viewport screenshots are unavailable in Blender background mode"}
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return {"available": False, "path": None, "reason": "No active Blender scene"}

    override = _view3d_override()
    if override is None:
        return {"available": False, "path": None, "reason": "No VIEW_3D area is available for screenshot capture"}

    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_path = _SCREENSHOT_DIR / f"vectra-loop-{iteration:02d}.png"
    original_filepath = scene.render.filepath
    try:
        scene.render.filepath = str(screenshot_path)
        with bpy.context.temp_override(**override):
            bpy.ops.render.opengl(write_still=True, view_context=True)
    except RuntimeError as exc:
        if screenshot_path.exists():
            screenshot_path.unlink(missing_ok=True)
        return {"available": False, "path": None, "reason": str(exc)}
    finally:
        scene.render.filepath = original_filepath

    return {"available": True, "path": str(screenshot_path), "reason": None}
