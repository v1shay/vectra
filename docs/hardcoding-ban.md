# Hardcoding Ban

Vectra must build scenes from user intent, scene state, model-generated planning, generic tools, spatial reasoning, and post-action observation. It must not make prompts look successful by routing to hidden recipes.

## Outlawed Runtime Patterns

- Prompt keyword routers that send a prompt to a domain-specific planner or builder.
- Named benchmark or demo scene builders in normal runtime.
- Domain templates such as maintenance bays, bedrooms, kitchens, offices, sofas, room shells, or focal furniture that create fixed object layouts.
- Automatic fallback scenes that replace failed model planning with canned geometry.
- Hidden fixture tools registered in the normal tool registry.
- Success statuses, screenshots, or demos that omit template/fallback/fixture provenance.
- Tests or docs that treat template output as evidence of general scene intelligence.

## Allowed Determinism

- Blender tool execution and schema validation.
- Spatial math, AABB bounds, anchors, collision/contact checks, and coordinate transforms.
- Observation snapshots, scene diffs, and prompt-obligation verification.
- Generic procedural tools when all scene meaning and parameters come from the model plan.
- Fixtures in tests or docs, as long as they cannot be imported, registered, or triggered by normal runtime.

## Required Runtime Standard

Every normal Vectra run must make its provenance visible:

- `planning_mode` should identify the organic planner path.
- `hardcoding_policy` must be `clean` for normal runtime.
- Metadata must show provider/model, validation retries, tool families, action count, and prompt obligations.
- Transcripts must explain intent, plan, selected actions, validation result, observed scene changes, repair reason, and next step.

If the model cannot produce a valid plan after bounded retry, Vectra must fail honestly. It must not create a canned scene to appear successful.

## Future Codex Pass Checklist

Before claiming a Vectra improvement:

- Search for prompt-specific routing, templates, fixture tools, fallback scenes, and benchmark builders.
- Confirm normal tool discovery excludes fixture or benchmark tools.
- Run anti-hardcoding prompts with negated domain words.
- Record action lists and metadata, not only screenshots.
- Treat hardcoded output as a failure unless the test explicitly targets a quarantined fixture.
