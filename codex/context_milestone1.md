# VECTRA STARTUP CONTEXT

## 1. CORE VISION

Vectra is a Claude Code for 3D inside Blender.

It is NOT:
- a chatbot
- a code generator
- a workflow macro system
- a set of hardcoded prompt handlers

It IS:
- an autonomous system that understands intent
- converts intent into structured plans
- executes real work inside Blender end-to-end

Non-negotiable:
- no hardcoded workflows
- no prompt-specific branching logic
- no raw Python execution from LLM

Vectra must:
- create, edit, animate, and repair scenes
- operate from text, images, and scene context
- execute tasks fully, not suggest them

---

## 2. SYSTEM ARCHITECTURE

Three-layer system:

Blender (execution)
↔ FastAPI runtime (planning boundary)
↔ LLM (decision layer)

Core rule:
Blender executes. Runtime thinks.

Pipeline:

User input
→ planner (LLM)
→ structured actions
→ ExecutionEngine
→ ToolRegistry
→ Blender

Forbidden:

LLM → raw Python → Blender

All execution must go through tools.

---

## 3. CURRENT STATE (IMPORTANT)

The execution kernel is COMPLETE and WORKING.

Working components:
- Blender add-on UI (Vectra panel)
- FastAPI backend
- HTTP bridge
- ToolRegistry (dynamic)
- ExecutionEngine (sequential, fail-fast, $ref support)
- Packaging + Blender integration
- Tests (passing)

Current tools:
- mesh.create_primitive
- object.transform

Execution pipeline works end-to-end:
Backend → actions → engine → Blender → real scene changes

---

## 4. CURRENT LIMITATION

The system has NO REAL INTELLIGENCE YET

- planner_stub is still active
- ignores prompt
- always returns same actions
- all prompts produce identical Blender behavior

This is the ONLY major missing piece.

---

## 5. TOOL SYSTEM

- tools are atomic
- tools are deterministic
- tools do NOT interpret prompts
- tools expose:
  - input_schema
  - (soon) output_schema

ToolRegistry:
- auto-discovers tools
- no engine changes needed to add tools

---

## 6. EXECUTION ENGINE

- sequential
- fail-fast
- resolves $ref dependencies
- validates params before execution
- returns ExecutionReport

STRICT RULE:
Only the ExecutionEngine can mutate Blender.

---

## 7. NEXT STEP (CRITICAL)

Replace planner_stub with real LLM planner.

Goal:

prompt + scene_state
→ LLM
→ structured actions
→ ExecutionEngine

Requirements:

- JSON-only output
- valid tool names only
- no hallucinated tools
- no raw Python
- validation before execution
- safe failure (return [])

Planner must be dynamic:
different prompts → different actions

---

## 8. SCENE STATE (UPGRADE NEEDED)

Current:
- active_object
- selected_objects
- current_frame

Target:
- full object list
- transforms
- selection flags

Needed for:
- “move cube”
- “edit existing object”
- scene-aware planning

---

## 9. TESTING

Already implemented:
- registry tests
- execution engine tests
- backend tests
- integration tests
- Blender headless validation

Next:
- planner tests (mock LLM)
- prompt → different action outputs
- Blender validation for different prompts

---

## 10. NON-NEGOTIABLE RULES

- NO raw Python from LLM
- NO backend bpy calls
- NO bypassing ExecutionEngine
- NO bypassing ToolRegistry
- NO hardcoded prompt logic
- Blender execution = main thread only
- tools = atomic only
- planner outputs structured actions ONLY

System boundary:

planner decides
Blender executes

---

## 11. FUTURE DIRECTION (IMPORTANT)

Vectra is NOT just a planner.

It will integrate:

### LLM (planning)
prompt → actions

### 3D models (geometry)
image/text → mesh
(Hugging Face, Shap-E, TripoSR, etc.)

### diffusion (textures)
prompt → materials

Final pipeline:

User input (text/image)
→ AI models (LLM + vision + generation)
→ structured actions / assets
→ ExecutionEngine
→ Blender

---

## 12. LONG-TERM SYSTEM

Vectra evolves into:

- Scene Intent Graph (SIG)
- Execution Graph (EG)
- critic + repair loop
- retry + replanning

Final system:

- autonomous
- scene-aware
- multimodal
- can create, edit, animate, repair
- no hardcoded workflows

---

## 13. CURRENT TASK

You are implementing:

LLM PLANNER

Specifically:
- planner.py
- llm_client.py
- replace planner_stub
- validate outputs
- ensure prompt changes behavior

SUCCESS CONDITION:

Different prompts produce different Blender results.

---

## FINAL LINE

Vectra is not here to help users use Blender.

Vectra is here to use Blender for the user.