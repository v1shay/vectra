# Vectra System Audit - 2026-04-08

This audit was produced after repeated live Blender failures on the docs ceiling prompt and follow-up regression testing during the Spatial Grounding v1 Phase A branch.

## Executive Summary

Vectra is not blocked on one bug. It is blocked on an interaction of four unfinished layers:

1. provider routing is unstable and can silently swap the reasoning model mid-run
2. scene observation and geometry feedback are too inaccurate for reliable spatial reasoning
3. the control loop rewards structural churn instead of scene quality
4. the tool surface and completion logic are still below the ceiling prompt's required modeling/composition level

The recent spatial grounding work did improve one narrow problem, raw coordinate guessing for simple grounding, but it was added into a runtime that still has weak observation, weak validation, weak completion control, and weak scene-level evaluation. That made the overall system more brittle because more automatic assumptions are now being layered on top of already inaccurate scene state.

## Live Evidence

- The docs explicitly say the ceiling prompt is the high-signal diagnostic for whether the system has real general 3D potential, not just narrow tool execution: [docs/vectra-single-run-diagnostic.md](/Users/agarwal/coding/vectra/docs/vectra-single-run-diagnostic.md#L1).
- The current-state matrix already says the key issues are still only `partial` or `open`, especially anti-stall, batching, composition, and audit pass rate: [docs/current-state-matrix.md](/Users/agarwal/coding/vectra/docs/current-state-matrix.md#L10).
- The latest stored audit artifact still shows `pass_rate: 0.0` and tags the ceiling prompt with `anti_stall_failure`, `scene_composition`, and `under_budget_completion_failure`: [.vectra/audits/20260407-203905/summary.json](/Users/agarwal/coding/vectra/.vectra/audits/20260407-203905/summary.json).
- That same artifact shows a timed-out run producing only a floor and a wall with no light, no camera, no animation, and no meaningful continuation: [.vectra/audits/20260407-203905/vectra-dev-01.json](/Users/agarwal/coding/vectra/.vectra/audits/20260407-203905/vectra-dev-01.json#L1).

## Findings

### P0. Provider fallback makes the Director non-deterministic and hard to debug

Vectra does not run one Director. It runs a candidate chain. `call_director()` will try the configured Director, then the controller model as `xai-director-fallback`, then local Ollama models, all behind the same interface: [agent_runtime/director/providers.py](/Users/agarwal/coding/vectra/agent_runtime/director/providers.py#L179), [agent_runtime/director/providers.py](/Users/agarwal/coding/vectra/agent_runtime/director/providers.py#L259).

The active runtime file confirms this chain is currently:

- `openai/gpt-5.4-mini`
- `x-ai/grok-4.1-fast`
- `qwen2.5-coder:32b`
- `deepseek-coder-v2:16b`

Source: [.vectra/runtime.env](/Users/agarwal/coding/vectra/.vectra/runtime.env).

This means the same prompt, same branch, and same UI can produce materially different behaviors depending on which fallback accepted the step. Your screenshot showing `xai-director-fallback` is direct proof that live runs are crossing models. This makes regression attribution noisy and can easily look like the system is "playing around" or "getting dumber" because, in effect, it is changing brains mid-task.

### P0. The Director reasons on inaccurate scene geometry

Live Blender objects can have real world-space bound boxes, but the observed `scene_state` stored for the Director does not use them. `build_scene_state()` uses `_approx_bounds()`, which only computes bounds from `location + dimensions` and ignores `matrix_world`/`bound_box`: [vectra/agent/observation.py](/Users/agarwal/coding/vectra/vectra/agent/observation.py#L19), [vectra/agent/observation.py](/Users/agarwal/coding/vectra/vectra/agent/observation.py#L167).

That inaccurate `bounds` payload is then treated as authoritative by `world_bounds()` whenever the object is represented as a mapping, which is exactly how the Director sees scene state: [vectra/tools/spatial.py](/Users/agarwal/coding/vectra/vectra/tools/spatial.py#L104).

As a result:

- rotated planes and walls can be reasoned about incorrectly
- scene footprint and extents can be wrong
- later placement decisions can be based on fake geometry
- anti-stall and composition scoring can be based on misleading structure growth

This is one of the most important bugs in the repo because it poisons both spatial grounding and control-layer reflection.

### P0. Progress scoring rewards junk scaffold growth and resets the anti-stall guard

The verifier treats created objects, added groups, parent changes, volume growth, and object-count increases as structural progress: [vectra/agent/reflection.py](/Users/agarwal/coding/vectra/vectra/agent/reflection.py#L122), [vectra/agent/reflection.py](/Users/agarwal/coding/vectra/vectra/agent/reflection.py#L165).

Then the Blender loop resets `ineffective_turns` whenever `structural_progress` is true, even if the scene is still ugly, open, incoherent, or moving backward relative to the prompt: [vectra/operators/run_task.py](/Users/agarwal/coding/vectra/vectra/operators/run_task.py#L438).

So the system is effectively rewarded for:

- adding more walls
- expanding scene footprint
- creating more primitives
- grouping objects

without checking whether:

- the room is actually enclosed
- the hero object exists
- the camera/light goals are being satisfied
- objects are inside the room
- the scene quality is improving

This explains why bad room scaffolds can survive for many turns, then suddenly die once they stop increasing cheap metrics.

### P1. Coordinated batch validation is too weak to guarantee useful batches

For broad prompts, the Director loop only rejects the case where there is exactly one actionable call. A batch of 2 to 4 actions is accepted even if all actions are local, repetitive, same-family, or semantically useless: [agent_runtime/director/loop.py](/Users/agarwal/coding/vectra/agent_runtime/director/loop.py#L317), [agent_runtime/director/loop.py](/Users/agarwal/coding/vectra/agent_runtime/director/loop.py#L466).

There is no validation that a broad scene batch must meaningfully advance any of:

- structure
- composition
- lighting
- camera
- animation
- composite-object construction

So the current "coordinated batch" rule checks batch size, not batch quality.

### P1. Resolver fallback is too assumption-heavy and hides model mistakes

`ReferenceResolver.resolve_target()` will silently fall back to the active object, first selected object, or most recently affected object when the model does not specify a valid target: [agent_runtime/director/resolver.py](/Users/agarwal/coding/vectra/agent_runtime/director/resolver.py#L165).

`resolve_location()` will also invent locations from:

- scene footprint for new primitives
- floor contact
- target-based offsets
- scene centroid/world origin

Source: [agent_runtime/director/resolver.py](/Users/agarwal/coding/vectra/agent_runtime/director/resolver.py#L261), [agent_runtime/director/resolver.py](/Users/agarwal/coding/vectra/agent_runtime/director/resolver.py#L331).

This keeps the loop moving, but it also converts weak or under-specified model output into plausible-looking bad actions instead of surfacing them as invalid. That is a major reason the system can keep constructing nonsense instead of failing fast.

### P1. Completion mode is mostly a prompt hint, not a real control policy

The loop computes `completion_mode_active` from turn budget state and passes it into the prompt: [agent_runtime/director/loop.py](/Users/agarwal/coding/vectra/agent_runtime/director/loop.py#L242), [agent_runtime/director/loop.py](/Users/agarwal/coding/vectra/agent_runtime/director/loop.py#L271).

But in the Blender runtime, the only concrete effect is a transcript line telling the model to prioritize a coherent finish: [vectra/operators/run_task.py](/Users/agarwal/coding/vectra/vectra/operators/run_task.py#L516).

There is no hard completion controller enforcing:

- finish now
- stop doing local polish
- satisfy missing prompt obligations in priority order
- declare failure if core goals are still missing near budget

That is why runs can meander and then die without a strong "close out the scene now" phase.

### P1. Spatial Phase A is deterministic but not hardened

The spatial tools directly set `target.location` and return success. They do not check collisions, containment, non-intersection, or whether the result is even inside the intended room footprint: [vectra/tools/spatial_tools.py](/Users/agarwal/coding/vectra/vectra/tools/spatial_tools.py#L115), [vectra/tools/spatial_tools.py](/Users/agarwal/coding/vectra/vectra/tools/spatial_tools.py#L165), [vectra/tools/spatial_tools.py](/Users/agarwal/coding/vectra/vectra/tools/spatial_tools.py#L215), [vectra/tools/spatial_tools.py](/Users/agarwal/coding/vectra/vectra/tools/spatial_tools.py#L262).

This is consistent with Phase A being intentionally minimal, but it means the spatial layer currently adds contact logic without any guardrails for:

- collision resolution
- containment in the room
- floor normalization beyond the first floor-like object
- preventing object placement outside the scene composition

That missing hardening is a gap, not a surprise, but it means the rest of the runtime needed to be stronger before this layer could safely carry more responsibility.

### P2. The modeling surface is still too primitive for the ceiling prompt

The primary creation tool is still `mesh.create_primitive`, backed by a small primitive set: [vectra/tools/mesh_tools.py](/Users/agarwal/coding/vectra/vectra/tools/mesh_tools.py#L14), [vectra/tools/mesh_tools.py](/Users/agarwal/coding/vectra/vectra/tools/mesh_tools.py#L37).

The diagnostic doc already says the ceiling prompt is testing real structure, composition, furniture/decor readability, lighting, camera framing, and visible animation in one run: [docs/vectra-single-run-diagnostic.md](/Users/agarwal/coding/vectra/docs/vectra-single-run-diagnostic.md#L33).

Right now the repo does not yet have enough modeling affordances or scene-specific control affordances to make that prompt routine. That is not the only blocker, but it is still a real blocker.

## Why The Spatial Branch Felt Worse

The spatial branch improved deterministic contact, but it landed into a system with:

- weak observed geometry
- weak scene-quality evaluation
- permissive assumption fallback
- no containment/collision hardening
- no hard completion controller

That combination makes a deterministic placement layer feel worse in practice, because the system now has more ways to confidently do the wrong thing.

The branch did not create the underlying control-layer weakness, but it exposed and amplified it.

## What Needs To Be Done First

### 1. Freeze the Director to one model chain for audits

For audit and regression testing:

- disable multi-family fallback
- run the ceiling prompt on exactly one Director model
- log provider/model per turn in the UI and audit artifacts

Do not compare branch quality while model roulette is still active.

### 2. Fix observed geometry before doing more spatial work

Replace `_approx_bounds()` with real world-space bounds from Blender `bound_box` and `matrix_world`, matching the logic already present for live Blender objects in [vectra/tools/spatial.py](/Users/agarwal/coding/vectra/vectra/tools/spatial.py#L62).

Until this is fixed, the Director is reasoning over fake room geometry.

### 3. Add objective-aware verification

The verifier needs to score prompt obligations, not just object churn. At minimum for broad scene prompts:

- floor exists
- enclosing structure exists
- at least one intentional composite object exists
- light exists and improves scene readability
- camera exists and frames the focal point
- animated presentation exists and is visibly meaningful
- placed objects stay within the intended scene footprint

Structural growth alone should not reset anti-stall.

### 4. Add real batch-quality validation

For broad prompts, validate not only batch size but also whether the batch materially advances the missing goals. Example checks:

- at least two distinct action families when the scene is underbuilt
- forbid repeated all-structure or all-transform batches once a scaffold already exists
- require camera/light/composite follow-through after structure milestones are reached

### 5. Reduce resolver assumptions on broad scene runs

The resolver should stop papering over weak model output in non-bootstrap cases. In particular:

- do not auto-select active objects as generic targets for broad scene layout
- do not auto-place new primitives from scene footprint unless the model truly omitted a location in a safe bootstrap context
- surface unresolved references as validation failures more often instead of "best effort" guesses

### 6. Make completion mode a controller, not a sentence

When `completion_mode_active` is true, the runtime should switch behavior:

- identify missing prompt obligations
- prioritize finish-critical tools
- block low-value local polish
- force a completion or explicit failure decision within the remaining turns

### 7. Only then do Spatial Phase B hardening

After the above, implement:

- AABB intersection detection
- bounded collision resolution
- strict raw-placement validation in non-empty scenes
- improved floor candidate detection and normalization

Doing this before the observation and evaluator layers are fixed will just make wrong decisions more deterministically wrong.

## Recommended Stabilization Order

1. lock audits to one Director model and remove silent fallback during evaluation
2. repair observed scene bounds and scene-state geometry fidelity
3. replace churn-based progress scoring with objective-aware scoring
4. strengthen batch validation for broad scene prompts
5. tighten resolver assumptions and fail fast on bad references
6. add a real completion controller
7. implement Spatial Phase B hardening
8. rerun the audit corpus and establish a new baseline artifact set

## Bottom Line

Vectra's hard part is still not done. The repo can execute tools, but the core intelligence infrastructure is still incomplete in exactly the areas that determine whether broad scene prompts feel smart:

- stable reasoning backend
- trustworthy scene observation
- objective-aware verification
- scene-level control and completion
- robust spatial hardening

Until those are fixed in that order, the system will continue to alternate between:

- doing nothing
- making a thin scaffold
- making ugly or incoherent local edits
- timing out or anti-stalling before a real finish
