# Spatial Intelligence V2

This pass gives Vectra a deterministic spatial substrate. It is not a scene template system, a layout planner, a visual critic, or an aesthetic model. It exists so the Director and Blender tools can reason from real geometry instead of object origins, approximate dimensions, active-object fallback, or scene-centroid guesses.

## What Changed

- `vectra/tools/spatial.py` is the shared geometry module for world-space AABB math.
- Scene observation now reports true world bounds through `world_bounds()`, plus per-object spatial metadata and stable scene anchors.
- Spatial placement tools use object bounds and derived anchors:
  - `object.place_on_surface`
  - `object.place_against`
  - `object.place_relative`
  - `object.align_to`
  - `object.place_at_anchor`
- The Director loop exposes compact spatial anchors in prompt context and tells the model to use spatial tools for contact and relative placement.
- The resolver preserves explicit tuple/list coordinates and turns earlier same-batch created object names into execution `$ref` values.

## Geometry Contract

All v2 spatial behavior is based on axis-aligned world bounds:

- `min` and `max`
- `center`
- `extents`
- `half_extents`
- face centers: `left`, `right`, `back`, `front`, `bottom`, `top`

For Blender objects, bounds come from `bound_box` transformed by `matrix_world` when available. Mapping records can also carry `world_bounds` or `bounds`. Invalid vectors, bool values, NaN, and infinity fail visibly instead of being coerced into placement.

## Relations

The v2 relation classifier emits deterministic records sorted by object name:

- `on`
- `against`
- `next_to`
- `left_of`
- `right_of`
- `in_front_of`
- `behind`
- `above`
- `below`

Relations are AABB/contact based. They are useful for grounding and adjacency, not for semantic design. Floor-like objects are intentionally filtered from noisy horizontal `next_to`/`against` records so the prompt does not fill with low-value floor adjacency.

## Anchors

Anchors are derived only from existing geometry:

- object face anchors: `Object.face.top`, `Object.face.front`, etc.
- floor anchors: `Floor.floor.top`, `Floor.floor.center`
- floor corners: `Floor.floor.corner.left.back`, etc.
- wall inner faces when inferable
- scene corners from current bounds

`object.place_at_anchor` can place an object at one of these anchors. For floor corners, it offsets the target inward by its half extents and grounds the bottom to the floor anchor. This supports generic corner placement without any room, bedroom, plant, or furniture template.

## Failure Behavior

Spatial tools are intentionally strict:

- missing targets fail
- missing references fail
- target equals reference fails
- unsupported surfaces, relations, or axes fail
- NaN, infinity, bool, and malformed vectors fail
- ambiguous strict object matches fail
- unresolved anchors fail

The general object tools may still use active-object fallback where that is useful for editing. Spatial relation tools do not, because picking the active object is how relation prompts silently become wrong.

## Current Limits

This is still AABB spatial intelligence, not full scene intelligence.

- No OBB support.
- No physics solver.
- No collision optimization.
- No full room solver.
- No semantic prompt obligation tracker.
- No visual critic.
- No asset/model import or generation.
- No composite builders.
- No LangGraph or delegated sub-agents.

The expected improvement is grounded, repeatable placement when the Director calls the right tools. It should make bad placements fail loudly and make good relation calls reliable. It does not guarantee beautiful finished scenes by itself.
