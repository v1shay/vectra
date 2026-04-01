from __future__ import annotations

from typing import Any


def summarize_scene_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
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

    summary_parts: list[str] = []
    if created:
        summary_parts.append(f"Created {', '.join(created)}")
    if moved:
        summary_parts.append(f"Moved {', '.join(item['name'] for item in moved)}")
    if removed:
        summary_parts.append(f"Removed {', '.join(removed)}")

    summary = "; ".join(summary_parts) if summary_parts else "No scene changes were detected."
    return {
        "created_objects": created,
        "removed_objects": removed,
        "moved_objects": moved,
        "summary": summary,
    }
