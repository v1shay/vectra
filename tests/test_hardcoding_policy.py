from __future__ import annotations

from pathlib import Path


FORBIDDEN_SOURCE_STRINGS = (
    "maintenance_bay",
    "generic_scene_composition",
    "GeneratedInterior",
    "GeneratedFocal",
    "scene.build_room_shell",
    "scene.build_focal_furniture",
    "skill.build_",
    "SceneIntent",
)


def test_active_runtime_source_has_no_forbidden_hardcoding_strings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_roots = [repo_root / "agent_runtime", repo_root / "vectra", repo_root / "scripts"]
    offenders: list[str] = []
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            content = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_SOURCE_STRINGS:
                if forbidden in content:
                    offenders.append(f"{path.relative_to(repo_root)} contains {forbidden}")

    assert offenders == []
