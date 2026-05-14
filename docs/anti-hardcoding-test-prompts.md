# Anti-Hardcoding Test Prompts

Purpose: catch template routing and benchmark leakage with minimal runtime cost.

Run each prompt in a mode that records provider/model, action list, fallback reason, planner id, and final object names. A pass requires no `maintenance_bay_htn_v1`, no `generic_scene_composition`, no `skill.build_*`, and no `GeneratedInterior` or `GeneratedFocal` objects unless the prompt explicitly asks for those exact names.

Normal runtime should report `planning_mode: organic_scene_graph_v1` and `hardcoding_policy: clean`.

## Tiny Shape Sanity

```text
Create exactly two different primitive shapes: a red cube on the left and a blue cylinder on the right. Leave space between them. Do not create a room, furniture, camera orbit, or maintenance bay.
```

Pass signal: two objects plus optional light/camera only. Fail if room shell, sofa, catwalk, workstation, or fallback template appears.

```text
Create a green cone balanced on top of a gray cube. The cone must touch the cube and must not float.
```

Pass signal: relation/contact works from the prompt. Fail if generic interior appears.

## Prompt Mutation Checks

```text
Build a repair kiosk with one tall service column, two small tool drawers, and a curved warning rail. Use simple shapes, but avoid catwalks, workstations, hazard stripes, and bay layouts.
```

Pass signal: creates kiosk-specific structure. Fail if maintenance-bay or generic room template appears.

```text
Make a small lunar sample table: flat tray, three labeled sample blocks, one magnifier arm, and a cool inspection light. No walls and no sofa.
```

Pass signal: table/tray/sample/magnifier semantics show up. Fail if room shell or focal sofa appears.

```text
Create a maintenance bay organically using generic tools: include a floor, raised access platform, three repair stations, cable paths, warning markings, lights, and a camera view. Do not use benchmark tools, templates, or any prebuilt maintenance-bay route.
```

Pass signal: low-level/procedural tools create prompt-specific objects. Fail if `skill.build_*`, benchmark metadata, or fixed maintenance-bay object names appear.

## Anti-Keyword Checks

```text
Make this scene coherent by arranging three existing objects into a neat diagonal display. Do not add any walls, furniture, or animation.
```

Pass signal: transforms existing objects only. Fail if the word `coherent` triggers generic composition.

```text
Add cinematic lighting to the existing objects only. Do not add geometry, do not add a room, and do not animate the camera.
```

Pass signal: light-only or camera framing changes. Fail if broad-scene fallback adds shell/furniture/orbit.

```text
Create a maintenance-bay warning sign only: one flat sign panel and two support posts. Do not create a maintenance bay, catwalk, workstations, cables, floor layout, or room.
```

Pass signal: only the sign and supports are created. Fail if any maintenance-bay route, benchmark object, or broad scene appears.

## Generalization Checks

```text
Create a tiny airport security tray scene: one tray, two shoes, a laptop rectangle, and three small bins. Use simple primitives and keep all objects grounded.
```

Pass signal: airport/tray/bin concepts are represented without templates.

```text
Create a compact kitchen counter corner with a sink basin, faucet, cutting board, and two mugs. Use only simple geometry. No sofa, no catwalk, no maintenance bay.
```

Pass signal: kitchen-specific objects appear. Fail if output resembles the known room-shell/focal-furniture fallback.

```text
Create a tiny observatory desk scene with a star chart, small telescope stand, notebook, and one warm desk light. Keep it grounded and readable from camera view.
```

Pass signal: generated plan reflects observatory/desk/telescope semantics without using a known room or furniture template.

## Reporting Template

For each run, record:

- prompt
- mode
- provider/model
- planner id
- fallback reason
- planning mode
- hardcoding policy
- action list
- final object names
- transcript excerpt
- screenshot path
- pass/fail reason
