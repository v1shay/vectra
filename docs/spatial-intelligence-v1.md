# Spatial Intelligence v1

Spatial v1 is a deterministic geometry layer for grounding and placement. It is not a layout planner, scene template system, prompt interpreter, visual critic, or aesthetic engine.

## What It Adds

- True world-space AABB observation through `world_bounds()`.
- Per-object spatial metadata:
  - center
  - extents and half-extents
  - face centers
  - grounded and floor-contact state
  - floor-like and wall-like geometry flags
- Deterministic relation records:
  - `on`
  - `against`
  - `next_to`
  - `left_of`
  - `right_of`
  - `in_front_of`
  - `behind`
  - `above`
  - `below`
- Generic anchors derived from current geometry:
  - object faces
  - floor top and center
  - wall inner faces when a thin vertical wall-like mesh is inferable
  - scene corners from existing bounds

All records are sorted by stable object names so repeated observation of identical input produces identical output.

## Placement Behavior

The existing placement tools remain the public surface:

- `object.place_on_surface`
- `object.place_against`
- `object.place_relative`
- `object.align_to`

They use AABB bounds and face/contact math rather than raw coordinate guessing. Side placement preserves the target object's existing coordinates on the non-contact axes, which allows deterministic multi-step placement such as placing an object against one wall and then another wall to reach a corner. Horizontal relative placement also preserves the target Z coordinate so objects stay grounded when they are moved beside another object.

## Live Blender Verification

These live tests verify whether spatial v1 is actually improving grounding and relative placement. They are not a proof of visual quality, semantic completion, model import, or aesthetic scene planning. The tests should be run from a clean Blender scene with the Vectra add-on loaded from the development repo.

### Preflight

1. Open Blender with the Vectra add-on enabled.
2. Set the Vectra development source path to the repo root.
3. Click `Start Backend` in the Vectra panel.
4. Confirm backend status becomes `online` and a backend log path is visible.
5. Start each test from an empty or intentionally reset scene.
6. Save the `.blend`, a screenshot, the task transcript, and the backend log after each prompt.

If the backend cannot start, the test fails before spatial validation. The UI should show `failed` and an actionable error instead of silently continuing.

### Runtime Reliability Check

Before judging the scene, check the transcript and backend log for provider attempts:

- A retryable HTTP `429`, `500`, `502`, `503`, `504`, timeout, network interruption, or JSON decode failure should record retry attempts.
- A retryable primary-provider failure should not end the task if fallback produces a valid actionable batch.
- HTTP `400`, `401`, `403`, and `404` should fail fast with explicit status metadata.
- The run should expose provider, model, status code when present, retry count or attempt number, elapsed time, failure reason, payload preview, selected provider, and final runtime state.

If a single retryable provider failure kills the loop while a fallback provider is configured, Pass A is not working.

### Test 1: Bed Against Back Wall

Prompt:

```text
Create a bed against the back wall.
```

Required output features:

- A floor-like surface exists or is created.
- A back wall-like vertical surface exists or is created.
- The bed object or primitive stand-in is grounded on the floor.
- The bed's rear face is flush or near-flush with the inner face of the back wall.
- The bed is not floating, not below the floor, and not visibly penetrating through the wall.
- The bed orientation reads as intentionally placed against the wall rather than randomly centered.

Expected spatial evidence:

- The bed object has `grounded: true` or non-null floor contact metadata.
- A relation record includes `against` between the bed and the back wall, or the bed face gap to the wall inner face is near zero within tolerance.

Failure signs:

- The bed appears at world origin, scene center, or a generic footprint fallback location away from the wall.
- The bed floats, clips deeply through the wall, or sits below the floor.
- The run creates only unrelated primitives and no identifiable main object.

Note: because this pass intentionally does not add composite builders, the "bed" may still be a simple primitive stand-in. This test verifies grounded wall placement, not bed visual fidelity.

### Test 2: Nightstand Beside Bed, Lamp On Nightstand

Prompt:

```text
Put a nightstand beside the bed and a lamp on the nightstand.
```

Run this immediately after Test 1 or from a scene that already contains a bed.

Required output features:

- A nightstand-like object is placed beside the bed, not on top of it and not far across the room.
- The nightstand is grounded on the floor.
- The nightstand has size-aware clearance from the bed: adjacent or slightly separated, but not intersecting.
- A lamp-like object is placed on the nightstand top face.
- The lamp is not on the floor, not floating above the nightstand, and not hidden inside the nightstand.

Expected spatial evidence:

- A relation record includes `next_to` between the nightstand and bed.
- A relation record includes `on` between the lamp and nightstand.
- The lamp bottom face is near the nightstand top face within tolerance.

Failure signs:

- The resolver silently uses the active object and moves the wrong object.
- The nightstand overlaps the bed or is placed at world origin.
- The lamp is grounded to the floor instead of the nightstand.
- The lamp uses raw coordinates that ignore the nightstand bounds.

### Test 3: Plant In Room Corner

Prompt:

```text
Put a plant in the corner of the room.
```

Required output features:

- A plant-like object or primitive stand-in is near a derived room or scene corner.
- The object is grounded on the floor.
- The object is inside the room footprint, not outside the walls.
- The object does not visibly penetrate through either wall.
- The object is not at the room center unless no corner can be derived.

Expected spatial evidence:

- Scene state exposes a corner anchor derived from existing floor and wall bounds.
- The plant center is close to that corner anchor in the horizontal axes.
- The plant has `grounded: true` or non-null floor contact metadata.

Failure signs:

- The plant goes to world origin or a scene-footprint fallback despite usable walls/floor.
- The plant floats, clips into a wall, or is placed outside the room.
- No corner anchor is derivable even though two wall-like surfaces and a floor-like surface exist.

### Test 4: Compact Interior Vignette Smoke Test

Prompt:

```text
Create a compact, coherent interior vignette with one clear focal point, a floor, two walls, one main object, one supporting object, a light, and a camera.
```

This is qualitative smoke coverage after Tests 1 through 3 pass. It should not be treated as a hard proof of aesthetic intelligence.

Required output features:

- Floor plus two wall-like surfaces are present.
- One main object is spatially readable as the focal object.
- One supporting object is placed in a grounded, related position near the main object.
- Light and camera objects exist.
- Objects stay inside the derived scene footprint when a footprint exists.
- Relative placement looks intentional: no obvious floating, deep clipping, or scattered unrelated primitives.

Failure signs:

- The scene is a pile of unrelated primitives with no grounded relationships.
- The model stops after partial structure and omits light or camera.
- Objects are placed mostly by origin or scene-footprint fallback despite existing anchors.
- The result depends on active-object accidents rather than explicit target/reference relations.

### Pass Criteria

Spatial v1 is working only if Tests 1 through 3 pass consistently across repeated clean runs. Test 4 should improve compared with the pre-spatial baseline, but it may still fail on beauty, completeness, or semantic richness because those capabilities are intentionally outside this pass.

If any required test fails, classify the failure before changing code:

- Provider/runtime failure: retries, fallback, or backend startup did not keep the task alive.
- Resolver failure: target/reference resolution used an ambiguous active-object or fallback path.
- Anchor failure: floor, wall, face, or corner anchors were not derived from existing geometry.
- Placement math failure: bounds were known but offsets/contact math produced floating, clipping, or wrong-side placement.
- Capability limit: the requested visual object needs composite builders, assets, or visual critique that v1 does not provide.

## Validation And Failure

Spatial placement fails visibly instead of inventing hidden behavior:

- missing target/reference values are invalid
- vague required spatial references do not silently fall back to the active object
- target and reference must resolve to different objects
- unsupported surfaces, relations, and axes are rejected
- bool, NaN, infinity, and negative offsets/distances are rejected

The resolver still supports ordinary active/selected-object fallback for non-spatial editing operations. Required spatial relation tools use stricter target/reference resolution because a bad relation is more harmful than a failed retry.

## Current Limits

- AABB only; no OBB, physics, constraints solver, or collision optimization.
- Geometry-derived anchors only; no prompt-specific object recipes or room templates.
- No composite object builders such as `create_bed`.
- No visual critic or aesthetic scoring.
- No prompt-obligation completion gate.
- No LangGraph, delegated sub-agents, learned spatial model, embeddings, or imported asset backend.

This layer improves grounding and relation-aware primitive placement, but it does not make broad scene prompts fully complete or beautiful by itself.
