Got it — this is your core persistent memory file.
You’ll drop this into your repo as something like:

docs/VECTRA_CORE.md

This is written to anchor the system, prevent drift, and enforce that Vectra becomes a true agentic system, not a demo tool.

⸻

📄 VECTRA CORE SYSTEM MEMORY (DO NOT BREAK THESE RULES)

⸻

0. 🔒 PURPOSE OF THIS DOCUMENT

This file defines:
	•	The true scope of Vectra
	•	The non-negotiable architectural rules
	•	The agentic capabilities required
	•	The design philosophy that must never be violated

If any implementation decision conflicts with this file →
the implementation is wrong, not the document

⸻

1. 🎯 WHAT VECTRA ACTUALLY IS

Vectra is:

A semantic operating system for 3D environments built inside Blender

It is NOT:
	•	a prompt tool
	•	a code generator
	•	a suggestion engine
	•	a passive assistant

⸻

Vectra MUST:
	•	Understand user intent (text + image + scene)
	•	Translate intent into structured representations
	•	Execute actions directly inside Blender
	•	Validate outputs
	•	Repair failures autonomously
	•	Complete tasks end-to-end without user intervention

⸻

Vectra MUST NOT:
	•	Suggest code and stop
	•	Output Blender scripts for the user to run
	•	Require manual execution steps
	•	Be limited to specific workflows (modeling, animation, etc.)

⸻

2. 🧠 CORE PRINCIPLE (ABSOLUTE)

Vectra completes tasks. It does not assist tasks.

⸻

3. 🧩 SYSTEM MODEL

⸻

3.1 Core Pipeline

User Input (text + image + scene)
        ↓
Multimodal Understanding
        ↓
Scene Intent Graph (SIG)
        ↓
Semantic Planner
        ↓
Execution Graph (EG)
        ↓
Execution Engine (Blender tools)
        ↓
Critic / Validation
        ↓
Repair Loop (if needed)
        ↓
Final Scene Output


⸻

4. 🧠 CORE REPRESENTATIONS

⸻

4.1 Scene Intent Graph (SIG)

Represents what the user wants
	•	Entities
	•	Attributes
	•	Relations
	•	Intent
	•	Constraints

SIG must:
	•	Be complete
	•	Be structured
	•	Be independent of execution

⸻

4.2 Execution Graph (EG)

Represents how to achieve the intent
	•	Ordered steps
	•	Tool calls
	•	Dependencies

EG must:
	•	Be executable deterministically
	•	Contain no ambiguity
	•	Be verifiable

⸻

5. ⚙️ EXECUTION PHILOSOPHY

⸻

CRITICAL RULE

No raw LLM output may directly control Blender

⸻

Execution MUST follow:

LLM → SIG → EG → Tool Registry → bpy Execution


⸻

NEVER:

LLM → Python → Blender


⸻

6. 🧠 AGENTIC CAPABILITY REQUIREMENTS

⸻

Vectra must be capable of all of the following simultaneously:

⸻

6.1 Creation
	•	Generate models from text
	•	Generate models from images
	•	Generate environments
	•	Generate materials

⸻

6.2 Editing
	•	Modify existing objects
	•	Adjust materials
	•	Change proportions
	•	Refactor scene structure

⸻

6.3 Animation
	•	Generate motion from prompts
	•	Apply motion to rigs
	•	Control timeline and camera
	•	Combine multiple animation layers

⸻

6.4 Repair
	•	Fix broken geometry
	•	Fix rigging issues
	•	Fix animation problems
	•	Resolve conflicts in scene

⸻

6.5 Execution
	•	Perform multi-step operations
	•	Handle dependencies
	•	Maintain scene consistency

⸻

6.6 Autonomy
	•	Detect failure
	•	Retry intelligently
	•	Re-plan partially
	•	Improve outputs iteratively

⸻

7. 🔁 AUTONOMOUS LOOP (MANDATORY)

Execute Step
   ↓
Evaluate Result
   ↓
If Failure:
   → Critic analyzes
   → Generate repair EG
   → Execute repair
   → Continue


⸻

Vectra must:
	•	Never silently fail
	•	Never stop mid-task without resolution
	•	Always attempt recovery

⸻

8. 🏗️ SYSTEM ARCHITECTURE (LOCKED)

⸻

Hybrid System

Blender Add-on (UI + execution)
        ↓
Local Runtime (FastAPI)
        ↓
Semantic + Planning System


⸻

8.1 Blender Add-on Responsibilities
	•	UI (side panel)
	•	Scene state extraction
	•	Tool execution
	•	Step visualization

⸻

8.2 Runtime Responsibilities
	•	SIG generation
	•	EG planning
	•	orchestration
	•	retry logic

⸻

8.3 Separation Rule

Blender executes. Runtime thinks.

⸻

9. 🧩 TOOL SYSTEM (STRICT)

⸻

Tools must be:
	•	Deterministic
	•	Validated
	•	Reversible (undo-safe)
	•	Modular

⸻

Tool Categories

Geometry
	•	create_mesh
	•	remesh
	•	boolean

Materials
	•	apply_shader
	•	generate_texture

Rigging
	•	auto_rig
	•	bind_mesh

Animation
	•	insert_keyframes
	•	path_animation

Scene
	•	camera control
	•	lighting

Repair
	•	fix_normals
	•	remove_nonmanifold

⸻

10. 🧪 TESTING REQUIREMENTS

⸻

Every layer must be testable:

⸻

SIG Tests
	•	schema validation
	•	completeness

⸻

EG Tests
	•	dependency correctness
	•	tool validity

⸻

Execution Tests
	•	Blender headless execution
	•	expected scene state

⸻

Golden Tests

Prompt → deterministic output

⸻

Stress Tests
	•	long task chains
	•	conflicting instructions
	•	failure injection

⸻

11. 🖥️ UI REQUIREMENTS (CLAUDE-CODE STYLE)

⸻

Vectra must behave like:

Claude Code for Blender

⸻

UI MUST:
	•	Exist as a Blender side panel
	•	Be embedded natively (NOT external window)
	•	Support:
	•	prompt input
	•	image upload
	•	task execution
	•	progress tracking
	•	history

⸻

UI MUST NOT:
	•	Be a suggestion-only interface
	•	Output code without execution
	•	Require user scripting

⸻

12. ⚠️ FAILURE MODES TO PREVENT

⸻

Vectra must NOT become:

⸻

❌ “Chatbot in Blender”
	•	Just prints responses

⸻

❌ “Code generator”
	•	Outputs scripts instead of executing

⸻

❌ “Workflow tool”
	•	Hardcoded pipelines

⸻

❌ “Demo system”
	•	Only works for specific examples

⸻

13. 🔒 NON-NEGOTIABLE RULES

⸻

	•	All actions go through SIG → EG
	•	No direct LLM → Blender execution
	•	No hardcoded workflows
	•	All outputs must be validated
	•	System must recover from failure
	•	UI must be native inside Blender
	•	System must complete tasks, not suggest them

⸻

14. 🚀 FINAL SYSTEM BEHAVIOR

⸻

User:

uploads image + types prompt

Vectra:
	•	understands intent
	•	builds semantic graph
	•	plans execution
	•	executes inside Blender
	•	detects errors
	•	repairs automatically
	•	outputs final result

⸻

15. 🧨 CORE IDENTITY

Vectra is:

An autonomous semantic execution system for 3D environments

⸻

16. 🔁 DEVELOPMENT GUARDRAIL

Before building ANY feature, ask:
	1.	Does this improve autonomy?
	2.	Does this reduce reliance on hardcoding?
	3.	Does this keep execution structured?
	4.	Does this maintain Blender-native control?
	5.	Does this complete tasks, not suggest them?

If any answer is “no” → rethink implementation.

⸻

17. 📌 FINAL LINE (DO NOT FORGET)

Vectra is not here to help users use Blender.
Vectra is here to use Blender for the user.

⸻

If you want next:

👉 I can now turn this into:
	•	￼ a SIG schema + validator (real code)
	•	￼ a planner v1 that actually generates EG
	•	and ￼ plug it into your current backend

That’s where the system becomes truly intelligent.