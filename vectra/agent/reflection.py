from __future__ import annotations

from typing import Any


def _bounds_volume(bounds: dict[str, Any]) -> float:
    minimum = bounds.get("min", [])
    maximum = bounds.get("max", [])
    if not (isinstance(minimum, list) and isinstance(maximum, list) and len(minimum) == 3 and len(maximum) == 3):
        return 0.0
    extents = [max(float(maximum[index]) - float(minimum[index]), 0.0) for index in range(3)]
    return extents[0] * extents[1] * extents[2]


def _visible_animation_names(scene_state: dict[str, Any]) -> list[str]:
    visible: list[str] = []
    for obj in scene_state.get("objects", []):
        if not isinstance(obj, dict):
            continue
        if bool(obj.get("visible_animation")):
            name = obj.get("name")
            if isinstance(name, str) and name.strip():
                visible.append(name)
    return sorted(visible)


def summarize_scene_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    execution_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_objects = {
        obj["name"]: obj
        for obj in before.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("name"), str)
    }
    after_objects = {
        obj["name"]: obj
        for obj in after.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("name"), str)
    }

    created = sorted(name for name in after_objects if name not in before_objects)
    removed = sorted(name for name in before_objects if name not in after_objects)

    moved: list[dict[str, Any]] = []
    changed: list[str] = []
    for name in sorted(set(before_objects).intersection(after_objects)):
        before_location = before_objects[name].get("location")
        after_location = after_objects[name].get("location")
        if (
            isinstance(before_location, list)
            and isinstance(after_location, list)
            and len(before_location) == 3
            and len(after_location) == 3
            and any(abs(float(after_location[index]) - float(before_location[index])) > 0.01 for index in range(3))
        ):
            moved.append(
                {
                    "name": name,
                    "before": [float(value) for value in before_location],
                    "after": [float(value) for value in after_location],
                }
            )
            continue

        for field_name in ("rotation_euler", "scale", "material_names", "parent", "collection_names", "keyframe_count", "light_energy", "camera_lens"):
            if before_objects[name].get(field_name) != after_objects[name].get(field_name):
                changed.append(name)
                break

    frame_changed = before.get("current_frame") != after.get("current_frame")
    before_groups = {
        group.get("name")
        for group in before.get("groups", [])
        if isinstance(group, dict) and isinstance(group.get("name"), str)
    }
    after_groups = {
        group.get("name")
        for group in after.get("groups", [])
        if isinstance(group, dict) and isinstance(group.get("name"), str)
    }
    added_groups = sorted(name for name in after_groups if name not in before_groups)
    parent_changes = sorted(
        name
        for name in set(before_objects).intersection(after_objects)
        if before_objects[name].get("parent") != after_objects[name].get("parent")
    )
    before_visible_animation = _visible_animation_names(before)
    after_visible_animation = _visible_animation_names(after)
    added_visible_animation = sorted(name for name in after_visible_animation if name not in before_visible_animation)
    before_volume = _bounds_volume(before.get("scene_bounds", {}))
    after_volume = _bounds_volume(after.get("scene_bounds", {}))
    volume_growth = max(after_volume - before_volume, 0.0)
    volume_growth_ratio = (volume_growth / before_volume) if before_volume > 0 else (1.0 if after_volume > 0 else 0.0)
    action_families: list[str] = []
    if isinstance(execution_payload, dict):
        metadata = execution_payload.get("metadata", {})
        if isinstance(metadata, dict):
            families = metadata.get("action_families", [])
            if isinstance(families, list):
                action_families = [value for value in families if isinstance(value, str)]

    summary_parts: list[str] = []
    if created:
        summary_parts.append(f"Created {', '.join(created)}")
    if moved:
        summary_parts.append(f"Moved {', '.join(item['name'] for item in moved)}")
    if changed:
        summary_parts.append(f"Adjusted {', '.join(changed)}")
    if removed:
        summary_parts.append(f"Removed {', '.join(removed)}")
    if frame_changed:
        summary_parts.append(f"Changed frame from {before.get('current_frame')} to {after.get('current_frame')}")
    if added_groups:
        summary_parts.append(f"Grouped objects into {', '.join(added_groups)}")
    if parent_changes:
        summary_parts.append(f"Updated parenting for {', '.join(parent_changes)}")
    if added_visible_animation:
        summary_parts.append(f"Added visible animation for {', '.join(added_visible_animation)}")

    progress_score = 0.0
    progress_reasons: list[str] = []
    if created:
        progress_score += min(len(created) * 0.8, 2.4)
        progress_reasons.append(f"created {len(created)} object(s)")
    if removed:
        progress_score += min(len(removed) * 0.3, 0.9)
        progress_reasons.append(f"removed {len(removed)} object(s)")
    if moved:
        progress_score += min(len(moved) * 0.45, 1.35)
        progress_reasons.append(f"repositioned {len(moved)} object(s)")
    if changed:
        progress_score += min(len(changed) * 0.3, 1.2)
        progress_reasons.append(f"adjusted {len(changed)} object(s)")
    if added_groups:
        progress_score += 0.9
        progress_reasons.append("added grouping")
    if parent_changes:
        progress_score += 0.75
        progress_reasons.append("changed parenting")
    if volume_growth_ratio >= 0.15:
        progress_score += 0.9
        progress_reasons.append("expanded scene structure")
    if len(after.get("objects", [])) > len(before.get("objects", [])):
        progress_score += 0.45
        progress_reasons.append("increased scene complexity")
    if len(after.get("lights", [])) > len(before.get("lights", [])):
        progress_score += 0.65
        progress_reasons.append("improved lighting setup")
    if after.get("active_camera") and not before.get("active_camera"):
        progress_score += 0.65
        progress_reasons.append("added camera framing")
    if added_visible_animation:
        progress_score += 1.4
        progress_reasons.append("added visible motion")
    elif after_visible_animation and frame_changed:
        progress_score += 0.8
        progress_reasons.append("advanced an animated scene")
    if any(family in {"layout", "structure", "light", "camera", "animation"} for family in action_families) and progress_score > 0.0:
        progress_score += 0.25
        progress_reasons.append("used a high-leverage action family")

    progress_score = round(progress_score, 3)
    structural_progress = bool(
        created
        or added_groups
        or parent_changes
        or volume_growth_ratio >= 0.15
        or len(after.get("objects", [])) > len(before.get("objects", []))
    )
    visible_animation = bool(after_visible_animation)
    meaningful_change = progress_score >= 0.75
    low_progress = 0.0 < progress_score < 1.5
    summary = "; ".join(summary_parts) if summary_parts else "No scene changes were detected."
    if progress_reasons:
        summary = f"{summary} Progress score {progress_score:.2f} ({', '.join(progress_reasons[:4])})."
    return {
        "created_objects": created,
        "removed_objects": removed,
        "moved_objects": moved,
        "changed_objects": changed,
        "frame_changed": frame_changed,
        "added_groups": added_groups,
        "parent_changes": parent_changes,
        "visible_animation_objects": after_visible_animation,
        "added_visible_animation_objects": added_visible_animation,
        "progress_score": progress_score,
        "progress_reasons": progress_reasons,
        "structural_progress": structural_progress,
        "visible_animation": visible_animation,
        "low_progress": low_progress,
        "meaningful_change": meaningful_change,
        "summary": summary,
    }
