# Spatial Live Test Plan

Use this document to verify whether the spatial layer is actually working in Blender. These tests are intentionally small. Do not trust the broad vignette prompt until the first three relation prompts pass.

## Setup

1. Open Blender with the Vectra panel visible.
2. Use `vectra-dev`.
3. Start from a fresh scene unless a test says otherwise.
4. Record the provider/model shown in the UI or backend log.
5. After each prompt, inspect the final scene in the viewport and, if needed, use the object transform panel to confirm positions.

If a provider/runtime error appears, classify the run as runtime failure first. Do not diagnose spatial behavior from an incomplete run.

## Test 1: Back-Wall Contact

Prompt:

```text
Create a bed against the back wall.
```

Required output features:

- A floor-like object exists.
- A wall-like object exists and is readable as a back wall.
- A bed-like object exists, even if made from simple primitives.
- The bed is grounded: bottom face at or near the floor top.
- The bed is against the wall: one bed side face is flush or near-flush with the wall inner face.
- The bed remains inside the floor footprint.
- No object required for the prompt is floating.

Hard fail signals:

- Bed appears at world origin without relation to wall.
- Bed intersects deeply through the wall.
- Bed floats above or sinks below the floor.
- The run reports missing required target/reference and stops before creating enough geometry.

What this verifies:

- true world bounds
- floor/wall likeness
- wall inner-face anchors or `object.place_against`
- no scene-centroid fallback for relation placement

## Test 2: Beside And On

Prompt:

```text
Put a nightstand beside the bed and a lamp on the nightstand.
```

Run this after Test 1 in the same scene.

Required output features:

- The nightstand is adjacent to the bed using object size, not object origins.
- There is visible clearance or near-contact between bed and nightstand.
- The nightstand remains grounded on the floor.
- The lamp bottom is flush or near-flush with the nightstand top.
- The lamp does not snap to the active object unless that object is actually the named nightstand.
- Parent/group operations, if used, reference the intended objects only.

Hard fail signals:

- Nightstand appears centered inside the bed.
- Lamp lands on the floor instead of on the nightstand.
- Lamp or nightstand target resolves to the wrong active object.
- Logs show duplicated or wrong child references during parenting.

What this verifies:

- `object.place_relative`
- `object.place_on_surface`
- strict target/reference resolution
- same-batch action refs for newly created named objects

## Test 3: Corner Placement

Prompt:

```text
Put a plant in the corner of the room.
```

Run this after Test 1, or start from a scene that already has a floor and two wall-like surfaces.

Required output features:

- The plant is grounded on the floor.
- The plant is near a derived floor or wall corner.
- The plant is inside the room footprint.
- The plant does not intersect deeply with either wall.
- The run uses existing geometry; it does not invent a hidden room template.

Hard fail signals:

- Plant appears in the center of the room when a corner is available.
- Plant is outside the floor footprint.
- Plant is embedded through the walls.
- The tool silently uses the active object as the plant or wall.

What this verifies:

- floor-corner anchors
- `object.place_at_anchor` or two `object.place_against` operations
- generic geometry-derived placement

## Test 4: Compact Vignette Smoke

Prompt:

```text
Create a compact, coherent interior vignette with one clear focal point. Build real structure, not random primitives: include a floor, at least one wall or other room-defining surface, and one intentional multi-part hero object that reads as designed furniture or decor. Keep the scene small enough to finish within the current turn budget. Arrange the scene so the focal point is obvious, reduce overlap, add at least one deliberate light, frame it with the camera, and add a short visible presentation animation using either the camera or a light. Finish only when the result feels like a staged corner of a room rather than a loose collection of shapes.
```

This is qualitative smoke coverage, not proof that the final intelligence layer exists.

Required output features:

- Room structure is grounded and spatially coherent.
- The focal object is composed of multiple related parts.
- Supporting objects do not spawn in nonsensical positions.
- At least one light exists and is intentionally placed.
- Camera placement is deliberate.
- At least one camera or light has visible keyframed motion.

Expected remaining limitations:

- The result may still look primitive.
- It may still lack strong aesthetics.
- It may still miss prompt obligations because there is no prompt checklist/completion gate.
- It may still fail if the provider produces unsupported tools or stops early.

## How To Report Failures

For every failed run, capture:

- exact prompt
- iteration count
- provider/model chain
- runtime state
- first validation error, if any
- final screenshot
- object names and transforms for the failed relation
- whether the failure was runtime, tool validation, spatial placement, or scene-quality/completion

Use this diagnosis:

- Runtime/provider failure: no complete actionable batch reached Blender.
- Tool-grounding failure: unsupported tool or schema-invalid arguments.
- Resolver failure: target/reference resolved to the wrong object or active object.
- Spatial failure: right objects were used but contact/grounding/anchor math was wrong.
- Capability failure: spatial placement worked, but the scene still lacked semantic completion, aesthetic composition, animation quality, or detailed modeling.
