# De-Hardcoding Implementation Evidence - 2026-05-14

Branch: `refactor/deterministic-model-layers`

## Removed From Runtime

- Maintenance-bay keyword route before the Director loop.
- Maintenance-bay HTN planner and benchmark validator modules.
- Registered `skill.build_*` maintenance-bay tools.
- Generic room/focal-furniture builder tools.
- Automatic `generic_scene_composition` fallback.
- Dormant legacy intent/construction pipeline modules and the old phase-1 live script.

## Current Runtime Standard

- Normal runs go through the Director loop and model-selected tools.
- Successful action batches include `planning_mode: organic_scene_graph_v1`.
- Successful action batches include `hardcoding_policy: clean`.
- Organic metadata records prompt obligations, semantic terms, planned scene nodes, task graph, and tool families.
- Panel transcripts now show intent, planning mode, obligations, plan, actions, task graph, tool families, provider/model, and validation retry status.

## Verification

Command:

```bash
.venv/bin/pytest
```

Result:

```text
169 passed
```

Focused coverage added or updated:

- Negated maintenance-bay prompts route through the Director, not a benchmark planner.
- Runtime tool discovery excludes `skill.build_*`, `scene.build_room_shell`, and `scene.build_focal_furniture`.
- Broad prompt failures retry or fail honestly instead of producing canned geometry.
- Organic metadata differs by prompt and selected tool batch.
- Active runtime source is scanned for forbidden hardcoding strings.
- Transcript output is readable and includes planning/runtime context.

## Live Blender Status

Not run in this pass. The next live check should run the prompts in `docs/anti-hardcoding-test-prompts.md` and capture provider/model, action list, transcript excerpt, final object names, and screenshots.
