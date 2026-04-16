from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised in plain Python tests
    bpy = None

from vectra.tools.spatial import (
    bounds_to_lists,
    spatial_anchors,
    spatial_metadata_for_object,
    spatial_relations,
    world_bounds,
)

_SCREENSHOT_DIR = Path(tempfile.gettempdir()) / "vectra-agent-screenshots"


def _vector_to_list(vector: Any, size: int = 3) -> list[float]:
    return [float(component) for component in vector[:size]]


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


def _animated_property_name(data_path: str) -> str:
    normalized = data_path.strip().lower()
    if normalized.startswith("location"):
        return "location"
    if normalized.startswith("rotation_euler"):
        return "rotation"
    if normalized.startswith("scale"):
        return "scale"
    if normalized.endswith("energy"):
        return "light_energy"
    if normalized.endswith("lens"):
        return "camera_lens"
    return normalized or "unknown"


def _animation_summary(obj: Any) -> dict[str, Any]:
    animation_data = getattr(obj, "animation_data", None)
    action = getattr(animation_data, "action", None)
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        return {
            "frame_start": None,
            "frame_end": None,
            "animated_properties": [],
            "max_property_deltas": {},
            "max_transform_delta": 0.0,
            "light_energy_delta": 0.0,
            "camera_lens_delta": 0.0,
            "visible_motion": False,
        }

    frame_start: float | None = None
    frame_end: float | None = None
    animated_properties: set[str] = set()
    max_property_deltas: dict[str, float] = {}
    max_transform_delta = 0.0
    light_energy_delta = 0.0
    camera_lens_delta = 0.0

    for curve in fcurves:
        keyframe_points = getattr(curve, "keyframe_points", [])
        if not keyframe_points:
            continue
        frames: list[float] = []
        values: list[float] = []
        for point in keyframe_points:
            co = getattr(point, "co", ())
            if len(co) < 2:
                continue
            frames.append(float(co[0]))
            values.append(float(co[1]))
        if not frames or not values:
            continue
        property_name = _animated_property_name(str(getattr(curve, "data_path", "")))
        animated_properties.add(property_name)
        frame_start = min(frame_start, min(frames)) if frame_start is not None else min(frames)
        frame_end = max(frame_end, max(frames)) if frame_end is not None else max(frames)
        delta = abs(max(values) - min(values))
        max_property_deltas[property_name] = max(delta, max_property_deltas.get(property_name, 0.0))
        if property_name in {"location", "rotation", "scale"}:
            max_transform_delta = max(max_transform_delta, delta)
        elif property_name == "light_energy":
            light_energy_delta = max(light_energy_delta, delta)
        elif property_name == "camera_lens":
            camera_lens_delta = max(camera_lens_delta, delta)

    visible_motion = bool(
        frame_start is not None
        and frame_end is not None
        and frame_end > frame_start
        and (
            max_transform_delta >= 0.25
            or light_energy_delta >= 25.0
            or camera_lens_delta >= 3.0
        )
    )
    return {
        "frame_start": int(frame_start) if frame_start is not None else None,
        "frame_end": int(frame_end) if frame_end is not None else None,
        "animated_properties": sorted(animated_properties),
        "max_property_deltas": max_property_deltas,
        "max_transform_delta": max_transform_delta,
        "light_energy_delta": light_energy_delta,
        "camera_lens_delta": camera_lens_delta,
        "visible_motion": visible_motion,
    }


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


def _attach_spatial_metadata(objects: list[dict[str, Any]]) -> None:
    floor_candidates = [
        obj
        for obj in objects
        if spatial_metadata_for_object(obj).get("is_floor_like")
    ]
    for obj in objects:
        obj["spatial"] = spatial_metadata_for_object(obj, floor_candidates=floor_candidates)

    relations = spatial_relations(objects)
    relations_by_source: dict[str, list[dict[str, Any]]] = {}
    for relation in relations:
        relations_by_source.setdefault(str(relation["source"]), []).append(relation)
    for obj in objects:
        obj["relations"] = relations_by_source.get(str(obj.get("name", "")), [])


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
        bounds = bounds_to_lists(world_bounds(obj))
        materials = _material_names(obj)
        keyframe_count = _keyframe_count(obj)
        animation_summary = _animation_summary(obj)
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
            "animation_summary": animation_summary,
            "visible_animation": bool(animation_summary.get("visible_motion")),
        }
        if getattr(obj, "type", "") == "LIGHT":
            object_record["light_energy"] = float(getattr(getattr(obj, "data", None), "energy", 0.0))
            object_record["light_type"] = str(getattr(getattr(obj, "data", None), "type", ""))
        if getattr(obj, "type", "") == "CAMERA":
            object_record["camera_lens"] = float(getattr(getattr(obj, "data", None), "lens", 0.0))
        objects.append(object_record)

    _attach_spatial_metadata(objects)

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
        "spatial_anchors": spatial_anchors(objects),
        "spatial_relations": spatial_relations(objects),
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
