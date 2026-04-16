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
