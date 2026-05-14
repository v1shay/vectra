# Hardcoding Audit - 2026-05-13

Scope: docs-only pass. Source was inspected for routing and evidence, but no source files were edited.

## Bottom Line

Both screenshot outputs have hardcoded or semi-hardcoded paths behind them.

- The maintenance-bay output is intentionally hardcoded benchmark machinery, not proof of general freeform scene intelligence.
- The coherent interior/basic-shape output can be produced by a generic composition fallback that hardcodes room shell, focal furniture, lighting, and camera-orbit actions when broad prompts fail validation.
- These paths are useful as scaffolding, but they must not be counted as evidence that Vectra can generally reason from arbitrary prompts.

## Hardcoded Systems Found

### 1. Maintenance-bay prompt router

Evidence:

- `agent_runtime/agent/reasoner.py:44` calls `plan_maintenance_bay_step(context)` before the freeform Director loop.
- `agent_runtime/intelligence/planner.py:41` routes prompts containing `maintenance` plus `bay`, `catwalk`, or `workstation`.
- `agent_runtime/intelligence/planner.py:56` builds a fixed maintenance-bay intent graph.
- `agent_runtime/intelligence/planner.py:111` decomposes that graph into fixed `skill.*` steps.
- `agent_runtime/intelligence/planner.py:236` explicitly says the planner "bypassed model-invented tool calls."

Assessment: hardcoded. This should be removed or quarantined behind an explicit benchmark mode. It should not trigger in normal `vectra-dev`.

### 2. Maintenance-bay deterministic skill tools

Evidence:

- `vectra/tools/maintenance_bay_tools.py:111` to `336` defines deterministic tools for floor, raised catwalk, workstation row, cable bundle, hazard stripes, overhead lights, and corridor camera.
- Object names, positions, dimensions, materials, counts, camera target, and lighting are fixed.

Assessment: hardcoded. These are acceptable only as fixture/building-block tests. They must not be used as a claimed general planner result.

### 3. Generic composition fallback batch

Evidence:

- `agent_runtime/director/loop.py:495` builds a fallback batch when broad-scene validation pressure is detected.
- The batch hardcodes `scene.build_room_shell`, `scene.build_focal_furniture`, `light.create`, and `animation.camera_orbit`.
- It hardcodes names, dimensions, light placement, light energy, camera animation frames, and target naming at `agent_runtime/director/loop.py:499` to `515`.
- `codex/docs/ai-runtime-root-cause-2026-05-13.md:37` to `45` records a provider run where the provider returned one tool call and Vectra returned the fallback batch instead.

Assessment: semi-hardcoded. It is less prompt-specific than maintenance bay, but it can make broad prompts look solved by returning the same starter composition.

### 4. Generic composition tools

Evidence:

- `vectra/tools/composition_tools.py:79` defines `scene.build_room_shell`.
- Defaults hardcode an 8 x 6 x 3.2 room shell, fixed wall/floor layout, names, and materials.
- `vectra/tools/composition_tools.py:147` defines `scene.build_focal_furniture`.
- It only supports `sofa`, `console`, and `sculpture`; each style is a fixed primitive arrangement.

Assessment: semi-hardcoded template system. Useful as atomic scene tools, but risky if automatically substituted for prompt reasoning.

### 5. Keyword broad-scene trigger

Evidence:

- `agent_runtime/director/loop.py:519` treats prompts containing broad keywords like `scene`, `room`, `interior`, `coherent`, `cinematic`, `focal point`, `lighting`, `camera`, `animate`, or `animation` as requiring coordinated batch behavior.
- This trigger can route many evaluation prompts into generic fallback territory.

Assessment: heuristic hardcoding. It is not prompt-specific, but it can bias tests toward the fallback system.

### 6. Style classifier for focal furniture

Evidence:

- `agent_runtime/director/loop.py:486` maps `console`, `desk`, `workstation`, or `table` to `console`; `sculpture`, `gallery`, `decor`, or `installation` to `sculpture`; everything else to `sofa`.

Assessment: small but real hardcoding. It should be replaced by model-selected structured choices or lower-level construction.

## What The Screenshots Most Likely Prove

### Screenshot 1

The prompt appears to be "Create a coh..." and the output shows a compact room-like structure, a focal object, light, and camera. This matches the generic composition fallback signature closely enough that it should be treated as fallback evidence until logs prove otherwise.

Minimum evidence needed before calling it freeform:

- provider/model chain
- parsed provider tool calls
- whether `generic_scene_composition` appears in metadata/logs
- final action list
- whether `scene.build_room_shell` or `scene.build_focal_furniture` was used

### Screenshot 2

The maintenance bay scene matches the deterministic benchmark path. It proves the Blender tool bridge can create geometry, not that the Director solved an arbitrary prompt.

Minimum evidence needed before calling it general:

- the prompt must avoid maintenance-bay trigger words
- no `planner: maintenance_bay_htn_v1`
- no `skill.build_*` maintenance-bay tools
- no `vectra_benchmark: maintenance_bay` objects

## Must Remove Or Quarantine

- Disable the maintenance-bay planner in normal `vectra-dev`; expose it only as an explicit benchmark/harness mode.
- Remove automatic generic-scene fallback from normal quality evaluation, or mark output status as `fallback_template_used` instead of `success`.
- Ensure UI transcript and backend metadata show when deterministic skills, fallback composition, or benchmark planners were used.
- Add audit scoring penalties for `scene.build_room_shell`, `scene.build_focal_furniture`, and `skill.build_*` unless the test is explicitly testing those tools.
- Add a run mode or env flag that disables all deterministic benchmark and generic fallback paths for anti-hardcoding tests.

## Evidence Standard

A Vectra run is not evidence of general scene intelligence unless all of these are true:

- No benchmark planner route.
- No generic fallback route.
- No prompt-specific skill tool.
- Provider/director action list contains varied choices tied to the exact prompt.
- Post-action observation confirms the required scene facts.
- Screenshot matches the prompt without relying on a fixed known template.
