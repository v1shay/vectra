# Vectra Single-Run Diagnostic

This file is the fastest high-signal test for the current state of Vectra.

Use one prompt, in one fresh scene, and judge the run against the failure map below. The goal is not to prove one narrow prompt works. The goal is to expose whether Vectra currently has real potential as a general 3D agent.

Right now Vectra is in a control-layer hardening stage, not a finished quality stage. That means the most honest ceiling prompt today should still demand real structure, composition, lighting, camera framing, and visible animation, but it should ask for a compact vignette that can plausibly finish inside the current budget.

## How To Use It

1. Open a fresh Blender scene with the Vectra panel visible.
2. Use `vectra-dev`.
3. Paste the exact prompt in the next section.
4. Let the run finish or visibly fail.
5. Compare what happened to the diagnosis rubric below.

## The Prompt To Paste

This is the current-stage ceiling prompt the system should be working toward right now:

```text
Create a compact, coherent interior vignette with one clear focal point. Build real structure, not random primitives: include a floor, at least one wall or other room-defining surface, and one intentional multi-part hero object that reads as designed furniture or decor. Keep the scene small enough to finish within the current turn budget. Arrange the scene so the focal point is obvious, reduce overlap, add at least one deliberate light, frame it with the camera, and add a short visible presentation animation using either the camera or a light. Finish only when the result feels like a staged corner of a room rather than a loose collection of shapes.
```

## Stretch Prompt

This is still the later-stage prompt Vectra should grow into after the current hardening pass:

```text
Create a coherent, cinematic interior scene with one clear focal point. Build real structure, not random primitives: include a floor, surrounding room elements, and at least one intentional multi-part object that reads as a designed piece of furniture or decor. Arrange the scene so the focal point is obvious, reduce overlap, add lighting that makes the scene readable and appealing, and frame it with the camera. Then add a short visible animation where either the camera or one of the lights moves across the scene in a way that improves the presentation. Finish only when the result feels intentionally composed rather than just populated with objects.
```

## Why This One Prompt Matters

A single successful run on this prompt exercises almost every system that currently matters:

- broad prompt actionability
- first-step execution
- scene construction
- composite object creation
- layout and composition
- lighting
- camera framing
- editing/refinement
- animation
- bounded completion

If Vectra cannot make meaningful progress here, it is still blocked on core control or capability issues, not just missing polish.

## What Counts As A Pass

Minimum pass:

- the run starts making real scene changes quickly
- the default cube is no longer the main output
- the result includes actual room structure, even if it is only a compact vignette
- there is at least one clearly intentional multi-part object
- the scene has a light and camera placement that look deliberate
- there is visible motion over time from either a light or the camera
- the system reaches a coherent stopping point instead of stalling

Strong pass:

- the scene has a clear focal hierarchy
- objects are arranged with purpose instead of scattered
- the composite object has proportions and part relationships that make sense
- the camera helps presentation instead of existing only because Blender starts with one
- lighting improves readability and mood
- the animation reads as presentation, not just a tiny transform change

## Fast Failure Diagnosis

### 1. Nothing meaningful happens at iteration 1

Symptoms:

- UI sits on something like `Observing and planning the next step...`
- scene stays at default cube, light, camera
- no real first action batch is executed

Likely issue:

- provider transport failure
- tool-call parse failure
- no-action Director output
- runtime state not being surfaced cleanly

Meaning:

The system is still blocked before Blender execution. This is an upstream runtime/actionability problem, not a scene-building problem.

### 2. It creates one or two primitives, then stalls or times out

Symptoms:

- floor only
- wall only
- one cube or one plane
- partial scaffold with no real continuation
- floor plus wall but no lighting, camera, hero object, or animation

Likely issue:

- completion control is still too weak after partial progress
- Director is not producing strong enough first batches
- batching is too weak
- composition follow-through is misfiring

Meaning:

The loop can act and may even start structure correctly, but it still cannot sustain construction through a finished scene.

### 3. It produces random blocks instead of a designed scene

Symptoms:

- repeated cubes or planes with weak relationships
- no object reads as intentional furniture or decor
- primitive spam replaces structure

Likely issue:

- composite-object capability is still too weak
- tool grounding is poor
- Director is underusing grouping, alignment, distribution, and proportions
- tool surface may still be below the required modeling ceiling

Meaning:

The runtime can place primitives, but it still cannot model intentionally.

### 4. It builds objects but the scene still looks flat or ugly

Symptoms:

- no clear focal point
- poor spacing
- overlap
- no useful camera framing
- light exists but does not improve the scene

Likely issue:

- scene composition reasoning is weak
- batch actions are too local
- editing/refinement logic is not strong enough
- observation summaries are not rich enough to drive better composition

Meaning:

The system can create things, but it cannot art-direct them yet.

### 5. It says it animated, but you cannot see motion

Symptoms:

- transcript mentions moving light or camera
- no visible playback change
- keyframes may exist but presentation does not change meaningfully

Likely issue:

- animation sequencing is weak
- motion span is too small
- animation verification is too shallow
- Director treats animation as a transform edit instead of a temporal presentation step

Meaning:

Animation is still nominal rather than real.

### 6. It keeps making tiny local edits and never finishes

Symptoms:

- repeated small moves
- tiny transforms
- single-object nudges
- no coherent final state

Likely issue:

- batching inefficiency
- anti-stall strategy switching is weak
- no strong completion mode
- Director is not prioritizing visible completion under budget

Meaning:

The system has local agency but poor task-level control.

## What This Prompt Diagnoses In One Run

If this prompt fails, you can usually place the system in one of these buckets immediately:

- `Upstream runtime blocked`
  - no first executable action
- `Construction loop weak`
  - starts but cannot continue to a finish
- `Modeling ceiling too low`
  - primitive spam instead of intentional objects
- `Composition layer weak`
  - objects exist but scene still looks bad
- `Animation layer weak`
  - static-looking result despite animation claims
- `Completion control weak`
  - endless small edits without a finished scene

## What This Prompt Does Not Prove

Even if it passes, it does not prove:

- robust mesh modeling
- advanced materials
- character animation
- high-end architectural precision
- production-quality art direction across many styles

It does prove the much more important thing right now:

whether Vectra is becoming a real general-purpose 3D agent instead of a fragile prompt-to-primitive toy.

## Recommendation

If this current-stage prompt cannot produce a coherent, intentionally composed, visibly animated vignette, the next priority should be stronger completion control, stronger coordinated batches, and better light/camera/animation follow-through before expanding into a much larger tool surface.
