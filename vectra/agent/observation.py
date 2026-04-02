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


def _material_names(obj: Any) -> list[str]:
    materials = getattr(getattr(obj, "data", None), "materials", None)
    if materials is None:
        active = getattr(obj, "active_material", None)
        return [active.name] if getattr(active, "name", None) else []
    return [
        material.name
        for material in materials
        if getattr(material, "name", None)
    ]


def _keyframe_count(obj: Any) -> int:
    animation_data = getattr(obj, "animation_data", None)
    action = getattr(animation_data, "action", None)
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        return 0
    total = 0
    for curve in fcurves:
        total += len(getattr(curve, "keyframe_points", []))
    return total


def _scene_centroid(objects: list[dict[str, Any]]) -> list[float]:
    if not objects:
        return [0.0, 0.0, 0.0]
    count = float(len(objects))
    return [
        sum(float(obj["location"][index]) for obj in objects) / count
        for index in range(3)
    ]


def _scene_bounds(objects: list[dict[str, Any]]) -> dict[str, list[float]]:
    if not objects:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    mins = [min(float(obj["bounds"]["min"][index]) for obj in objects) for index in range(3)]
    maxes = [max(float(obj["bounds"]["max"][index]) for obj in objects) for index in range(3)]
    return {"min": mins, "max": maxes}


def _collection_groups(scene: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for collection in getattr(getattr(scene, "collection", None), "children_recursive", []):
        object_names = [
            obj.name
            for obj in getattr(collection, "objects", [])
            if getattr(obj, "name", None)
        ]
        if object_names:
            groups.append({"name": collection.name, "object_names": object_names})
    return groups


def build_scene_state(context: bpy.types.Context) -> dict[str, Any]:
    if bpy is None:
        raise RuntimeError("Blender Python API is unavailable")
    scene = context.scene
    active_object = context.active_object
    objects = []
    scene_objects = getattr(scene, "objects", [])
    for obj in scene_objects:
        location = _vector_to_list(obj.location)
        rotation = _vector_to_list(obj.rotation_euler)
        scale = _vector_to_list(obj.scale)
        dimensions = _vector_to_list(getattr(obj, "dimensions", (2.0, 2.0, 2.0)))
        bounds = _approx_bounds(obj)
        materials = _material_names(obj)
        keyframe_count = _keyframe_count(obj)
        object_record = {
            "name": obj.name,
            "type": obj.type,
            "selected": bool(obj.select_get()),
            "active": active_object is not None and obj == active_object,
            "location": location,
            "rotation_euler": rotation,
            "scale": scale,
            "dimensions": dimensions,
            "bounds": bounds,
            "parent": getattr(getattr(obj, "parent", None), "name", None),
            "children": [child.name for child in getattr(obj, "children", [])],
            "collection_names": [
                collection.name
                for collection in getattr(obj, "users_collection", [])
                if hasattr(collection, "name")
            ],
            "material_names": materials,
            "has_animation": keyframe_count > 0,
            "keyframe_count": keyframe_count,
        }
        if getattr(obj, "type", "") == "LIGHT":
            object_record["light_energy"] = float(getattr(getattr(obj, "data", None), "energy", 0.0))
            object_record["light_type"] = str(getattr(getattr(obj, "data", None), "type", ""))
        if getattr(obj, "type", "") == "CAMERA":
            object_record["camera_lens"] = float(getattr(getattr(obj, "data", None), "lens", 0.0))
        objects.append(object_record)

    active_camera = getattr(scene, "camera", None)
    lights = [
        {
            "name": obj["name"],
            "type": obj.get("light_type"),
            "energy": obj.get("light_energy", 0.0),
            "location": obj["location"],
        }
        for obj in objects
        if obj.get("type") == "LIGHT"
    ]
    centroid = _scene_centroid(objects)
    bounds = _scene_bounds(objects)

    return {
        "active_object": active_object.name if active_object else None,
        "selected_objects": [obj.name for obj in context.selected_objects],
        "current_frame": scene.frame_current,
        "active_camera": active_camera.name if active_camera is not None else None,
        "scene_centroid": centroid,
        "scene_bounds": bounds,
        "groups": _collection_groups(scene),
        "lights": lights,
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
