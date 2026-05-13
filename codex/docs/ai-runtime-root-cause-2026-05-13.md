# AI Runtime Root Cause - 2026-05-13

## User-visible failure

The Blender panel could sit on `Runtime State: awaiting_model_response` for minutes, then fail with:

`The provider produced unsupported or invalid tool calls: ... needs a coordinated batch of 2 to 4 tool calls rather than a single local action.`

The API provider was not simply offline. Live requests reached the configured provider and returned model output, but broad scene prompts often came back as a single local action or as a static scene batch that omitted requested animation.

## Root cause

1. Broad prompts such as "create a coherent cinematic interior scene" need multiple coordinated actions.
2. The Director validator correctly rejected single-action broad responses.
3. The old path handled that rejection by asking the provider for a corrective retry.
4. Provider calls commonly took several seconds each, and repeated validation retries, backend deadlines, and Blender bridge timeouts made the UI appear stuck in `awaiting_model_response`.
5. For animation prompts, a static 4-call batch could be accepted even when it missed the requested animation, leaving the output short of the user intent.

The clean maintenance-bay screenshot was real Blender geometry, not a rendered fake, but it came from the deterministic maintenance-bay planner/tool path rather than proving the freeform model loop was healthy.

## Fix

- Added generic composition tools:
  - `scene.build_room_shell`
  - `scene.build_focal_furniture`
- Added `animation.camera_orbit` for a real camera keyframed move.
- Added a fast Director fallback for broad prompts when the provider returns a schema-valid but too-small batch.
- Added explicit animation validation: prompts that ask for animation must include an animation-family action.
- The fallback now stops the agent loop after producing a complete starter scene, avoiding another immediate provider call.
- Fixed the generic sofa geometry so the current spatial verifier no longer reports floating or unsupported parts.
- Updated local live-validation scripts to launch `agent_runtime.main:app` from the repo root instead of brittle `main:app` from inside `agent_runtime`.

## Live evidence

Provider-only probe:

- Prompt: cinematic interior scene with focal point, room elements, lighting, camera framing, and short camera animation.
- Result: HTTP 200.
- Elapsed: 6.76 seconds.
- Provider/model: `openai-director` using `nvidia/nemotron-3-nano-30b-a3b`.
- Parsed provider tool count: 1.
- Director fallback: `generic_scene_composition`.
- Validation retry: false.
- Returned actions: `scene.build_room_shell`, `scene.build_focal_furniture`, `light.create`, `animation.camera_orbit`.
- Continue loop: false.

Live Blender run:

- Backend: temporary uvicorn server using `agent_runtime.main:app`.
- Blender: background packaged add-on execution.
- Result phase: `success`.
- Iterations: 1.
- Mesh count: 9.
- Light count: 1.
- Camera action: `CameraAction`.
- Camera action frame range: 1 to 72.
- Spatial verifier status: no `floating` or `unsupported` issues in the final status.
- Progress score: 6.20.

Full automated tests:

- `.venv/bin/pytest`
- Result: 169 passed.
