# Vectra Intelligence Layer Audit - 2026-04-16

This audit reviews the current codebase against the intended Vectra goal: a fast, agentic Blender tool that can create coherent, beautiful scenes with structure, animation, character or asset models, lighting, backgrounds, repair, delegated sub-agents, imported model backends, and LangGraph-style orchestration.

## Verdict

Vectra is currently a Level 2 intelligence layer.

It has moved beyond a stub planner. It has a real single-agent Director loop, provider adapters, tool schemas, validated tool execution, scene observation, basic reflection, a Blender-native UI, and an audit harness. That is meaningful infrastructure.

It is not yet a semantic scene intelligence layer. The current agent does not build a durable Scene Intent Graph, does not compile an Execution Graph, does not delegate to sub-agents, does not use LangGraph, does not attach real visual inputs to model calls, does not import or generate high-detail models, and does not have a hard quality/completion controller.

The strongest honest description is:

> A single Director-model tool orchestrator with deterministic Blender tools and basic loop reflection.

The end-state target is:

> A graph-orchestrated, multi-agent, multimodal 3D production system with asset/model generation, semantic scene planning, objective quality gates, and autonomous repair.

## Level Scale

| Level | Name | Meaning | Current status |
| --- | --- | --- | --- |
| 0 | Stub/demo | Hardcoded prompt handling or fixed actions | Passed |
| 1 | Execution kernel | Validated tool registry, action execution, Blender mutation | Passed |
| 2 | Single-agent Director | One model chooses bounded tool batches with observation and retries | Current |
| 3 | Self-correcting scene agent | Objective-aware planning, verification, completion, and repair | Not yet |
| 4 | Graph intelligence layer | LangGraph or equivalent orchestration with delegated specialist agents | Not yet |
| 5 | Production scene creator | Imports/generates assets, characters, materials, lighting, animation, backgrounds, and cinematic composition reliably | Not yet |

## What Exists Today

The runtime is split cleanly:

- Blender add-on and execution live under `vectra/`.
- FastAPI runtime lives under `agent_runtime/main.py`.
- The agent path is `/agent/step`, using `AgentService`, `reason_step`, and `DirectorLoop`.
- Tool execution is validated and sequential through `vectra/execution/engine.py`.
- Tool discovery is dynamic through `vectra/tools/registry.py`.
- Provider transport is separated from Director control through `agent_runtime/director/adapters.py` and `agent_runtime/director/providers.py`.
- Scene observation and reflection live in `vectra/agent/observation.py` and `vectra/agent/reflection.py`.
- The live audit corpus is in `vectra/audit/corpus.py`.

Important current behaviors:

- Provider candidates can include the configured Director, xAI fallback, Ollama primary, and Ollama secondary (`agent_runtime/director/providers.py:179`).
- The Director validates tool names and arguments before execution (`agent_runtime/director/loop.py:378`).
- Broad prompts trigger pressure toward a 2-4 action batch, but only through a shallow single-action rejection rule (`agent_runtime/director/loop.py:317`, `agent_runtime/director/loop.py:466`).
- `vectra-code` mode can expose a bounded Python escape hatch (`agent_runtime/director/loop.py:424`), although normal `vectra-dev` mode is tool-only.
- Current memory is disabled by default, in-memory when enabled, and Chroma is a stub (`agent_runtime/memory/manager.py:27`, `agent_runtime/memory/providers/chroma.py:8`).

## What It Can Strongly Achieve Now

Vectra can strongly achieve these tasks when the model emits compatible tool calls:

- Create and transform simple primitive objects: cube, cylinder, cone, torus, plane, UV sphere, ico sphere (`vectra/tools/mesh_tools.py:14`).
- Execute validated tool batches with `$ref` dependencies between actions.
- Duplicate, delete, select, parent, distribute, and align objects.
- Create floor-like grounding and AABB-based placement relations such as on, against, beside, above, and aligned.
- Apply basic material parameters: RGB base color, roughness, metallic.
- Create and adjust lights and camera.
- Insert simple transform keyframes for object animation.
- Detect some visible animation by keyframe span and transform/light/camera deltas (`vectra/agent/observation.py:119`).
- Surface provider/runtime states in the Blender UI instead of hiding all failures as generic waiting.
- Catch unsupported tool names and schema-invalid arguments, then do one corrective retry.
- Run a local Python test suite successfully.

Verification run from this audit:

```text
.venv/bin/python -m pytest
94 passed in 0.55s
```

Notes:

- `python -m pytest` failed because `python` is not on PATH.
- `python3 -m pytest` failed because that interpreter lacks `pytest`.
- `.venv/bin/python -m pytest` is the correct local command.
- `pyright` is not installed in PATH or `.venv/bin`.
- `blender` is not installed in PATH, so no fresh live Blender audit was run in this pass.

## What It Cannot Do Yet

Vectra cannot yet reliably do the things the final intelligence layer must do:

- It cannot reliably turn a broad scene prompt into a coherent, beautiful, finished scene.
- It cannot produce character models, rigged characters, meaningful character animation, or custom mesh assets.
- It cannot import or generate 3D model assets from a model backend.
- It cannot generate texture maps or rich materials beyond basic Principled BSDF settings.
- It cannot use image understanding in the actual model request. The current prompt includes screenshot metadata and a file path, not image bytes or an image content part (`agent_runtime/director/loop.py:277`).
- It cannot delegate work to specialist sub-agents.
- It cannot use LangGraph or any other graph runtime today. A repo search only finds LangGraph as a future extension point in docs.
- It cannot build a durable SIG/EG pipeline. The current Director prompt explicitly tells the model not to emit plans, intents, entity graphs, recipes, or precomputed construction pipelines (`agent_runtime/director/prompts.py:61`).
- It cannot enforce final scene quality before accepting `task.complete`.
- It cannot visually judge beauty, focal hierarchy, lighting quality, composition, overlap, camera framing, or animation readability.
- It cannot run deterministic model audits while fallback routing can swap the active model mid-run.

## Major Bugs And Limitations

### P0. The SIG/EG Architecture Is Not Active

The core docs describe the intended pipeline as:

```text
User input -> Scene Intent Graph -> Execution Graph -> Tool Registry -> Blender
```

The current runtime does not do that. The old `SceneIntent` pipeline still exists in files, but `extract_scene_intent()` now raises a retirement error (`agent_runtime/llm_client.py:78`), and `build_scene_pipeline()` can only catch that error (`agent_runtime/scene_pipeline.py:100`).

This means Vectra is currently prompt -> Director tool calls -> resolver -> actions, not prompt -> SIG -> EG -> tools.

This is the central gap between Level 2 and Level 4.

### P0. Visual Feedback Is Metadata, Not Vision

The Blender add-on captures viewport screenshots, but the Director prompt only receives:

```json
{"available": true, "path": "...", "reason": null}
```

That is not multimodal perception. The model is not seeing the pixels. It is seeing a local path string. This prevents reliable composition, lighting, camera, and "beautiful scene" feedback.

### P0. Completion Is Not Quality-Gated

`task.complete` is accepted when it has a non-empty summary (`agent_runtime/director/loop.py:397`). The Blender loop then marks success when the response says complete or `continue_loop` is false (`vectra/operators/run_task.py:474`).

There is no objective verifier requiring:

- floor or structure exists
- hero object exists
- lighting exists and improves readability
- camera frames the focal point
- animation is visible
- prompt-specific obligations are satisfied

This means a model can declare completion too early and the loop can accept it.

### P0. Live Verification Script Is Stale

`scripts/verify_phase1_live.py` still references `planner_module.extract_intent` and `planner_module.plan_actions` (`scripts/verify_phase1_live.py:186`), but `agent_runtime/planner.py` no longer exposes those names. That script will fail before it reaches its intended verification path.

The unit suite passes, but this stale live script is a real tooling bug.

### P0. Observed Bounds Are Approximate

`build_scene_state()` uses `_approx_bounds()` (`vectra/agent/observation.py:19`) and assigns those bounds for every object (`vectra/agent/observation.py:184`). That uses object location and dimensions rather than world-space `bound_box` plus `matrix_world`.

The spatial helper has stronger logic in `world_bounds()`, but the observed scene state handed to the Director is less faithful. Rotated walls, transformed planes, and non-axis-aligned objects can be misrepresented.

### P1. Provider Fallback Makes Audits Non-Deterministic

The Director candidate chain includes primary Director, xAI fallback, Ollama primary, and Ollama secondary (`agent_runtime/director/providers.py:179`). This is useful for resilience, but it makes quality audits noisy because the "same" run may use different reasoning models.

The latest stored live audit used only `ollama-primary:qwen2.5-coder:32b`, timed out, and had `pass_rate: 0.0` in `.vectra/audits/20260407-203905/summary.json`.

For real regression testing, provider/model must be frozen or at least treated as a test dimension.

### P1. Batch Validation Checks Size, Not Quality

For broad prompts, the current validator rejects exactly one non-observe action (`agent_runtime/director/loop.py:466`). A batch with 2-4 low-value actions passes even if it does not advance structure, focal hierarchy, lighting, camera, or animation.

This is why the system can look busy without becoming more intelligent.

### P1. Resolver Assumptions Hide Model Mistakes

The resolver falls back to active object, first selected object, recent history, scene footprint, floor contact, target-derived positions, or scene centroid (`agent_runtime/director/resolver.py:201`, `agent_runtime/director/resolver.py:277`, `agent_runtime/director/resolver.py:372`).

That keeps execution moving, but it can convert bad model output into plausible wrong actions. For a broad scene agent, some missing references should become validation failures instead of automatic guesses.

### P1. Progress Scoring Rewards Object Churn

Reflection adds progress for created objects, groups, parenting, volume growth, object count growth, lighting, camera, and animation (`vectra/agent/reflection.py:122`). Structural progress resets ineffective turns (`vectra/operators/run_task.py:438`).

That is better than a stub anti-stall check, but it still rewards scaffold growth more than prompt satisfaction or visual scene quality.

### P1. Memory Is Not Real Long-Term Intelligence

Memory is disabled unless `VECTRA_AGENT_MEMORY_ENABLED` is set (`agent_runtime/memory/manager.py:27`). The Chroma provider is a reserved stub that returns no results (`agent_runtime/memory/providers/chroma.py:20`).

Current memory cannot support long-running projects, style persistence, learned scene patterns, asset history, or user preferences.

### P1. Asset And Model Generation Are Missing

The current modeling surface is primitive- and modifier-based. It can block out a scene, but it cannot produce:

- character meshes
- rigged models
- detailed furniture
- procedural props
- imported asset packs
- generated backgrounds
- model-from-image outputs
- texture maps

This is why "beautiful real scenes" is not achievable yet except in a very limited primitive-art sense.

### P2. `vectra-code` Is A Useful Escape Hatch, But Not Final Architecture

`vectra-code` exposes `python.execute_blender_snippet` only in code mode (`agent_runtime/director/loop.py:424`). The executor blocks several unsafe patterns, but this still bypasses the final SIG/EG principle unless it is treated as a debug-only capability or wrapped in a structured, reviewed code-execution node.

The final agent should prefer structured tools and generated assets over direct Python snippets.

### P2. Scene Prompt Context Is Too Compact

The Director prompt compactly lists only the first 24 objects (`agent_runtime/director/loop.py:154`). It includes name, type, active/selected state, location, dimensions, parent, materials, and animation count, but not rich semantic labels, per-object bounds, visible screen position, occlusion, focal role, or material/lighting contribution.

This is not enough context for robust art direction.

## What The Current Agent Should Demonstrably Fail

These should be expected failures today and become regression tests for the intelligence-layer buildout:

- "Create a cinematic interior with a focal object, styled materials, background, and camera animation." Current likely outcome: partial primitives, weak completion, no coherent art direction.
- "Create a rigged character walking through a lit environment." Current likely outcome: impossible or primitive substitute; no character/rig pipeline.
- "Use this reference image to recreate the room layout." Current likely outcome: no real image understanding; screenshot/image path only.
- "Import a sofa model, cloth curtains, and a window background." Current likely outcome: no asset import/model registry path.
- "Delegate environment, character, lighting, and animation work to separate agents and merge results." Current likely outcome: no sub-agent graph or merge protocol.
- "Run a 40-step scene build with rollback after failed model import." Current likely outcome: no durable graph state, rollback, or asset failure recovery.
- "Judge whether the final shot is beautiful and improve it." Current likely outcome: no visual critic; only structural reflection.

## End-State Capabilities To Build Toward

By the end of the intelligence layer creation, Vectra should be able to:

- Parse user intent into a Scene Intent Graph with entities, attributes, relations, style, constraints, composition goals, and animation goals.
- Compile that SIG into an Execution Graph with ordered dependencies, tool/model calls, asset imports, checks, and repair branches.
- Run a LangGraph-style orchestrator with typed nodes for planner, scene architect, asset/model generator, layout solver, material artist, lighting/camera director, animator, critic, and repair.
- Delegate subtasks to specialist sub-agents and merge their outputs through a shared scene graph.
- Use imported model backends for meshes, characters, props, textures, and backgrounds.
- Attach real images/screenshots to model calls for visual reasoning.
- Maintain long-term memory for user preferences, scene assets, style decisions, and successful construction patterns.
- Gate completion through objective and visual rubrics.
- Produce audit artifacts that explain which agent did what, which model/backend was used, what failed, what was repaired, and why the scene passed.

## How To Test Core Capabilities

### 1. Always Run Unit Tests

Use the repo virtualenv:

```bash
.venv/bin/python -m pytest
```

This currently passes all 94 tests and should remain the minimum bar.

### 2. Fix And Restore The Live Verification Script

Before relying on `scripts/verify_phase1_live.py`, update it away from retired planner APIs. It should verify the current Director loop, not the old intent planner.

Minimum checks:

- backend health
- add-on package freshness
- tool action validation
- provider runtime-state visibility
- one simple create/edit run in Blender
- one invalid-tool rejection
- one broad prompt that produces a coordinated batch

### 3. Run The Live Audit Corpus

After provider configuration and Blender path are known:

```bash
.venv/bin/python -m vectra.audit.runner --launch-backend --modes vectra-dev --limit 1 --timeout-seconds 240
```

Then expand to the full corpus:

```bash
.venv/bin/python -m vectra.audit.runner --launch-backend --modes vectra-dev --timeout-seconds 240
```

For model comparison, run one model family at a time. If provider fallback remains enabled, include provider chain in the pass/fail analysis.

### 4. Use The Single-Run Ceiling Prompt

Run `docs/vectra-single-run-diagnostic.md` in a fresh Blender scene. It should remain the fastest manual test for whether the agent is graduating from "tool executor" to "scene builder."

Record:

- provider/model chain
- actions per turn
- object count
- whether a floor/room shell exists
- whether a hero object reads as intentional
- whether light/camera are deliberate
- whether animation is visible
- whether completion was justified

### 5. Add Capability-Specific Golden Tests

Add live or mocked tests for:

- primitive creation
- object editing
- spatial grounding
- composite furniture/decor
- lighting plus camera framing
- visible animation
- cleanup/repair
- unsupported asset request fail-closed behavior
- screenshot/image-aware prompt once multimodal input is implemented

Each test should assert final scene state, not just tool call existence.

### 6. Add Future Intelligence-Layer Tests

When building the new layer, add tests that fail today:

- SIG schema completeness for broad prompts.
- EG dependency correctness and retry branches.
- LangGraph node trace: planner -> architect -> assets -> layout -> materials -> lighting -> animation -> critic -> repair.
- Delegated sub-agent outputs merging without name collisions.
- Imported model backend with a mocked model registry.
- Visual critic receives actual image content and rejects poor composition.
- Completion is blocked until objective obligations are satisfied.

## Recommended Build Order

1. Repair live verification scripts and keep `.venv/bin/python -m pytest` green.
2. Add true world-space bounds to `build_scene_state()`.
3. Add a prompt-obligation evaluator that detects floor, structure, hero object, light, camera, visible animation, and completion readiness.
4. Gate `task.complete` through the evaluator.
5. Replace shallow batch-size validation with objective-aware batch validation.
6. Add audit mode that freezes the provider/model chain.
7. Add real multimodal screenshot/image content to provider requests.
8. Reintroduce SIG and EG as first-class runtime objects.
9. Add LangGraph-style orchestration and specialist sub-agent nodes.
10. Add asset/model import providers, then character/rig/material/background generation.
11. Add a visual critic and repair loop that can reject ugly or incomplete scenes.

## Bottom Line

Vectra is no longer a toy execution kernel. It is a working Level 2 agent loop with real tools, validation, provider abstraction, Blender execution, and a passing unit suite.

The intelligence layer is still missing the parts that will make it feel genuinely smart: semantic scene graphs, objective-aware control, real vision, delegated sub-agents, imported models, graph orchestration, asset generation, and visual quality gates.

The next milestone should not be "more prompts." It should be the first real semantic intelligence layer: SIG -> EG -> graph orchestration -> verified Blender execution.
