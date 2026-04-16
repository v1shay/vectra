# Vectra Current-State Matrix

This matrix rebases the April 2026 audits against the current code instead of treating the older findings as perfectly current.

Latest full intelligence-layer audit: `docs/intelligence-layer-audit-2026-04-16.md`.

## Status Key

- `patched`: the original root cause has been directly addressed in code
- `partial`: the failure mode is still possible, but the original root cause is no longer the whole story
- `open`: the issue is still materially present

## Audit Rebaseline

| Audit finding | Status | Current code basis |
| --- | --- | --- |
| Provider/runtime transport and actionability stalls on broad prompts | `partial` | Transport is now adapter-specific in `agent_runtime/director/adapters.py` and `agent_runtime/director/providers.py`, tool schemas are no longer duplicated in structured-tool prompts, and runtime states are surfaced through `vectra/operators/run_task.py`. Broad prompts can still fail, but the original transport diagnosis is no longer the full explanation. |
| Anti-stall layer kills legitimate in-progress builds | `partial` | Progress evaluation is now weighted in `vectra/agent/reflection.py`, and the Blender loop now treats structural progress differently in `vectra/operators/run_task.py`. This should reduce false stalls, but it still needs live validation against the ceiling prompt and audit suite. |
| Modeling surface too weak for composite objects | `partial` | `vectra/tools/mesh_tools.py` now includes `cylinder`, `cone`, `torus`, and `ico_sphere`, and `vectra/tools/modifier_tools.py` adds constrained modifiers. Composite modeling is still limited, but the previous primitive ceiling is no longer as low. |
| Tool grounding mismatch between Director reasoning and live tool surface | `partial` | The Director now validates tool names and arguments before execution in `agent_runtime/director/loop.py`, including one corrective retry. This should stop unsupported affordances from reaching Blender, but prompt quality and capability mapping still need live evaluation. |
| Scene editing solved with micro-actions instead of scene-level batches | `partial` | The Director loop now rejects certain single-action batches for broad scene/edit/animation prompts in `agent_runtime/director/loop.py`. This creates stronger batch pressure, but the art-direction quality of those batches still needs further tuning. |
| Animation treated as nominal keyframing instead of visible motion | `partial` | Scene observation now records animation spans and visible-motion signals in `vectra/agent/observation.py`, and reflection scores visible animation in `vectra/agent/reflection.py`. The system can now verify more than “a keyframe exists,” but more animation tools may still be needed. |
| Blender addon UI hides runtime/provider failures as ordinary work-in-progress | `patched` | The addon surfaces runtime states directly in `vectra/operators/run_task.py` and `vectra/ui/panel.py`, and provider timeout/deadline metadata now flows up from the adapter and provider layers. |
| Audit still shows pass rate 0 and composition/completion failures dominate | `open` | Historical audit artifacts still show failure. The corpus now includes the docs ceiling prompt in `vectra/audit/corpus.py`, but the suite must be rerun after this hardening pass to produce a new baseline. |

## What Still Matters Most

1. Treat the current system as Level 2: a single-agent Director loop with deterministic tools, not yet the final semantic intelligence layer.
2. Repair `scripts/verify_phase1_live.py`; it still references retired planner APIs and should verify the current Director path.
3. Run the updated audit suite, especially the docs ceiling prompt under `vectra-dev`, with provider/model chain recorded per run.
4. Confirm that structural partial progress improves scene completion instead of only rewarding object churn.
5. Check whether batch pressure improves scene composition or just changes the failure shape.
6. Build the missing SIG/EG, visual-feedback, objective-completion, LangGraph/sub-agent, and imported-model layers before judging Vectra as a real scene-production agent.
